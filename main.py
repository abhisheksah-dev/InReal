import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from googlesearch import search
from newspaper import Article
from pydantic import BaseModel

app = FastAPI(title="Improved Fact Checker API", version="2.0.2")

templates = Jinja2Templates(directory="templates")


class FactCheckRequest(BaseModel):
    claim: str
    max_results: Optional[int] = 5


class Evidence(BaseModel):
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    sentiment: str  # 'supporting', 'contradicting', 'neutral', 'context'
    credibility_score: float
    publication_date: Optional[str] = None


class FactCheckResult(BaseModel):
    claim: str
    verdict: str  # e.g., "Fact-Checkable", "Subjective Opinion", "Speculation"
    accuracy_score: Optional[float] = None  # 0-100, null for non-factual
    confidence: str  # 'high', 'medium', 'low', 'not applicable'
    summary: str
    detailed_analysis: str
    evidence: List[Evidence]
    timestamp: str
    sources_analyzed: int


class RealFactChecker:
    def __init__(self):
        # initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.model = None  # type: ignore
            print("Warning: GEMINI_API_KEY not found. Using fallback analysis.")

        # news sources
        self.trusted_sources = {
            "thehindu.com": 0.9,
            "indianexpress.com": 0.85,
            "timesofindia.indiatimes.com": 0.8,
            "hindustantimes.com": 0.8,
            "business-standard.com": 0.85,
            "theprint.in": 0.85,
            "scroll.in": 0.8,
            "thewire.in": 0.8,
            "newslaundry.com": 0.85,
            "altnews.in": 0.95,
            "boomlive.in": 0.9,
            "nature.com": 0.95,
            "sciencemag.org": 0.95,
            "who.int": 0.9,
            "icmr.gov.in": 0.9,
            "reuters.com": 0.95,
            "apnews.com": 0.95,
            "bbc.com": 0.9,
        }

        self.rate_limiter = asyncio.Semaphore(50)  # 50 QPM

    async def _classify_claim(self, claim: str) -> dict:
        """Use Gemini to classify the claim before fact-checking."""
        if not self.model:
            # Basic fallback classification
            if "?" in claim or any(
                q in claim.lower() for q in ["what is", "who is", "why"]
            ):
                return {
                    "classification": "nonsensical",
                    "reason": "The input is a question, not a declarative claim.",
                }
            subjective_words = ["best", "worst", "greatest", "love", "hate", "beautiful"]
            if any(word in claim.lower() for word in subjective_words):
                return {
                    "classification": "subjective",
                    "reason": "The claim uses subjective language, making it a matter of opinion.",
                }
            return {"classification": "factual", "reason": "Proceeding with fact-check."}

        try:
            prompt = f"""
            Analyze the following claim and classify it. Your primary goal is to determine if it is a verifiable factual statement, an opinion, or something else.

            CLAIM: "{claim}"

            Respond with a single, minified JSON object with two keys:
            1. "classification": Choose ONE of the following strings: "factual", "subjective", "speculation", "nonsensical", "harmful".
            2. "reason": A brief, user-facing explanation for your classification.

            Examples:
            - Claim: "The Eiffel Tower is 330 meters tall." -> {{"classification": "factual", "reason": "This is a specific, objective claim that can be verified with evidence."}}
            - Claim: "That politician is the greatest in history." -> {{"classification": "subjective", "reason": "This is a value judgment that depends on personal criteria and cannot be objectively proven true or false."}}
            - Claim: "By 2050, humanity will have a base on Mars." -> {{"classification": "speculation", "reason": "This is a prediction about a future event that cannot be verified at present."}}
            - Claim: "why is the sky blue" -> {{"classification": "nonsensical", "reason": "This is a question, not a verifiable claim."}}
            """
            async with self.rate_limiter:
                response = await self.model.generate_content_async(prompt)
            return self.extract_json(response.text)
        except Exception as e:
            print(f"Error in claim classification: {str(e)}")
            return {
                "classification": "factual",
                "reason": "Error during classification, attempting standard fact-check.",
            }

    async def search_web(self, query: str, max_results: int = 10) -> List[dict]:
        """Performs a web search using the googlesearch library."""
        executor = ThreadPoolExecutor(max_workers=4)
        try:
            loop = asyncio.get_event_loop()
            raw_urls = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: list(
                        search(query, num_results=max_results, lang="en",)
                    ),
                ),
                timeout=15,
            )
            tasks = [self.extract_article_content(url) for url in raw_urls]
            results = await asyncio.gather(*tasks)
            return [res for res in results if res]
        except (Exception, asyncio.TimeoutError) as e:
            print(f"Search error: {str(e)}")
            return []

    async def extract_article_content(self, url: str) -> Optional[dict]:
        """Extract article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()

            # Skip articles with minimal content
            if len(article.text) < 200:
                return None

            domain = urlparse(url).netloc.lower().replace("www.", "")
            return {
                "title": article.title or "No title available",
                "url": url,
                "snippet": (
                    (article.text or article.meta_description or "")[:300] + "..."
                ),
                "source": domain,
                "full_text": article.text,
                "publication_date": (
                    article.publish_date.isoformat() if article.publish_date else None
                ),
            }
        except Exception:
            return None

    def calculate_credibility_score(self, source: str) -> float:
        domain = source.lower()
        if domain in self.trusted_sources:
            return self.trusted_sources[domain]
        for trusted_domain, score in self.trusted_sources.items():
            if trusted_domain in domain:
                return score
        if any(indicator in domain for indicator in [".edu", ".gov", ".org"]):
            return 0.7
        if any(
            indicator in domain for indicator in ["blog", "forum", "wordpress", "tumblr"]
        ):
            return 0.3
        return 0.5

    def extract_json(self, text: str) -> dict:
        """Extracts the first valid JSON object from a string."""
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Fallback for malformed JSON or plain text responses
        print(f"Warning: Could not parse JSON from response: {text}")
        return {}

    async def analyze_evidence_for_factual_claim(
        self, claim: str, evidence_text: str
    ) -> dict:
        """Analyzes evidence for a factual claim."""
        if not self.model:
            # Fallback analysis for factual claims
            return {
                "sentiment": "neutral",
                "relevance_score": 0.5,
                "summary": "Gemini API not configured. Cannot perform detailed analysis.",
            }
        try:
            prompt = f"""
            Analyze the evidence provided in relation to the claim.

            CLAIM: "{claim}"

            EVIDENCE: "{evidence_text[:7000]}"

            Respond with a single, minified JSON object with three keys:
            1. "sentiment": String. Must be one of "supporting", "contradicting", or "neutral".
            2. "relevance_score": Float between 0.0 and 1.0. How relevant is the evidence to the claim?
            3. "summary": String. A one-sentence summary of what the evidence says about the claim.
            """
            async with self.rate_limiter:
                response = await self.model.generate_content_async(prompt)
            return self.extract_json(response.text)
        except Exception as e:
            print(f"Gemini evidence analysis error: {str(e)}")
            return {
                "sentiment": "neutral",
                "relevance_score": 0.0,
                "summary": "Error during analysis.",
            }

    async def generate_final_summary(
        self, claim: str, verdict: str, evidence: List[Evidence]
    ) -> dict:
        """Generates the final summary and detailed analysis using Gemini."""
        if not self.model:
            return {
                "summary": "Basic analysis complete.",
                "detailed_analysis": "Detailed analysis requires Gemini API.",
            }

        evidence_summary = "\n".join(
            [
                f"- Source: {e.source} (Credibility: {e.credibility_score:.2f}, Sentiment: {e.sentiment}): {e.title}"
                for e in evidence[:15]
            ]
        )

        prompt = f"""
        You are a fact-checking analyst. Your task is to synthesize the provided information into a final report.

        CLAIM: "{claim}"
        INITIAL VERDICT: This claim has been classified as "{verdict}".

        EVIDENCE GATHERED:
        {evidence_summary}

        TASK:
        Based on all the information, generate a final analysis. Respond with a single, minified JSON object with two keys:
        1. "summary": A concise, one-paragraph summary for a general audience. If the claim is subjective or speculation, explain that. If it's factual, state the conclusion (e.g., mostly true, mostly false, mixed).
        2. "detailed_analysis": A longer, more nuanced analysis. Elaborate on the key evidence, mention the credibility of sources, and discuss any complexities, different perspectives, or important context.
        """
        try:
            async with self.rate_limiter:
                response = await self.model.generate_content_async(prompt)
            return self.extract_json(response.text)
        except Exception as e:
            print(f"Final summary generation error: {str(e)}")
            return {
                "summary": "Error generating summary.",
                "detailed_analysis": "Could not generate detailed analysis due to an error.",
            }

    def calculate_accuracy_and_confidence(
        self, evidence: List[Evidence]
    ) -> tuple[Optional[float], str]:
        """Calculates accuracy score and confidence level for factual claims."""
        supporting = [e for e in evidence if e.sentiment == "supporting"]
        contradicting = [e for e in evidence if e.sentiment == "contradicting"]

        if not supporting and not contradicting:
            return None, "low"

        supporting_weight = sum(
            e.credibility_score * e.relevance_score for e in supporting
        )
        contradicting_weight = sum(
            e.credibility_score * e.relevance_score for e in contradicting
        )
        total_weight = supporting_weight + contradicting_weight

        if total_weight == 0:
            return 50.0, "low"

        accuracy = (supporting_weight / total_weight) * 100

        # Confidence calculation
        total_evidence_count = len(supporting) + len(contradicting)
        avg_credibility = (
            sum(e.credibility_score for e in supporting + contradicting)
            / total_evidence_count
        )
        if total_evidence_count >= 5 and avg_credibility > 0.7:
            confidence = "high"
        elif total_evidence_count >= 3 and avg_credibility > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return accuracy, confidence

    async def fact_check(self, claim: str, max_results: int = 5) -> FactCheckResult:
        """Main fact-checking workflow."""
        # 1. Classify the claim
        classification_result = await self._classify_claim(claim)
        verdict = classification_result.get("classification", "factual").title()
        initial_reason = classification_result.get(
            "reason", "No reason provided."
        )

        if verdict in ["Nonsensical", "Harmful"]:
            raise HTTPException(status_code=400, detail=f"Invalid Claim: {initial_reason}")

        # 2. Handle non-factual claims (Subjective/Speculation)
        if verdict in ["Subjective", "Speculation"]:
            query = f'perspectives on "{claim}"'
            search_results = await self.search_web(query, max_results)
            evidence_list = [
                Evidence(
                    title=r["title"],
                    url=r["url"],
                    snippet=r["snippet"],
                    source=r["source"],
                    relevance_score=1.0,  # Context is always relevant
                    sentiment="context",
                    credibility_score=self.calculate_credibility_score(r["source"]),
                    publication_date=r.get("publication_date"),
                )
                for r in search_results
            ]
            evidence_list.sort(key=lambda x: x.credibility_score, reverse=True)
            final_summary = await self.generate_final_summary(
                claim, verdict, evidence_list
            )

            return FactCheckResult(
                claim=claim,
                verdict=verdict,
                accuracy_score=None,
                confidence="not applicable",
                summary=final_summary.get("summary", initial_reason),
                detailed_analysis=final_summary.get(
                    "detailed_analysis",
                    "This claim is not fact-checkable. The following sources provide context.",
                ),
                evidence=evidence_list,
                timestamp=datetime.now().isoformat(),
                sources_analyzed=len(evidence_list),
            )

        # 3. Handle Factual Claims
        fact_check_query = f'fact check "{claim}"'
        queries = [claim, fact_check_query]
        search_results = []
        for q in queries:
            search_results.extend(await self.search_web(q, max_results))

        # Deduplicate results by URL
        unique_results = {r["url"]: r for r in search_results}.values()

        analysis_tasks = [
            self.analyze_evidence_for_factual_claim(claim, r["full_text"])
            for r in unique_results
        ]
        analyses = await asyncio.gather(*analysis_tasks)

        evidence_list = []
        for result, analysis in zip(unique_results, analyses):
            if not analysis:
                continue
            evidence_list.append(
                Evidence(
                    title=result["title"],
                    url=result["url"],
                    snippet=analysis.get("summary", result["snippet"]),
                    source=result["source"],
                    relevance_score=analysis.get("relevance_score", 0.0),
                    sentiment=analysis.get("sentiment", "neutral"),
                    credibility_score=self.calculate_credibility_score(
                        result["source"]
                    ),
                    publication_date=result.get("publication_date"),
                )
            )

        # Sort evidence by a combined score
        evidence_list.sort(
            key=lambda x: x.relevance_score * x.credibility_score, reverse=True
        )

        accuracy, confidence = self.calculate_accuracy_and_confidence(evidence_list)
        final_summary = await self.generate_final_summary(
            claim, verdict, evidence_list
        )

        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            accuracy_score=accuracy,
            confidence=confidence,
            summary=final_summary.get("summary", "Analysis complete."),
            detailed_analysis=final_summary.get("detailed_analysis", ""),
            evidence=evidence_list,
            timestamp=datetime.now().isoformat(),
            sources_analyzed=len(evidence_list),
        )


# Initialize fact checker
fact_checker = RealFactChecker()


@app.get("/")
async def root():
    return {
        "message": "Improved Fact Checker API - Now with claim classification",
        "version": "2.0.3",
        "endpoints": {"ui": "/web", "api_check": "/fact-check", "docs": "/docs"},
    }


@app.post("/fact-check", response_model=FactCheckResult)
async def check_fact(request: FactCheckRequest):
    """
    Fact-checks a claim by first classifying it and then gathering evidence.
    - **Factual claims** get an accuracy score.
    - **Subjective claims** get a summary of perspectives.
    - **Invalid claims** are rejected.
    """
    if not request.claim or not request.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")
    if len(request.claim) > 500:
        raise HTTPException(
            status_code=400, detail="Claim is too long (max 500 characters)."
        )

    return await fact_checker.fact_check(request.claim, request.max_results)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_api_status": "available" if fact_checker.model else "not_configured",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/web", response_class=HTMLResponse)
async def web_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/web", response_class=HTMLResponse)
async def web_fact_check(
    request: Request, claim: str = Form(...), max_results: int = Form(5)
):
    try:
        if not claim or not claim.strip():
            return templates.TemplateResponse(
                "index.html", {"request": request, "error": "Claim cannot be empty."}
            )
        if len(claim) > 500:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Claim is too long (max 500 characters)."},
            )

        result = await fact_checker.fact_check(claim, max_results)
        return templates.TemplateResponse(
            "result.html", {"request": request, "result": result}
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": f"API Error: {e.detail}"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": f"An unexpected error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    if not os.getenv("GEMINI_API_KEY"):
        print(
            "Warning: GEMINI_API_KEY environment variable not set. The API will run with limited, fallback functionality."
        )

    uvicorn.run(app, host="0.0.0.0", port=8000)
