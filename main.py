import json
import os
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from googlesearch import search
from newspaper import Article
from pydantic import BaseModel

app = FastAPI(title="Real Fact Checker API", version="1.0.0")

class FactCheckRequest(BaseModel):
    claim: str
    max_results: Optional[int] = 5

class Evidence(BaseModel):
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    sentiment: str  # 'supporting', 'contradicting', 'neutral'
    credibility_score: float
    publication_date: Optional[str] = None

class FactCheckResult(BaseModel):
    claim: str
    accuracy_score: float  # 0-100
    confidence: str  # 'high', 'medium', 'low'
    summary: str
    detailed_analysis: str
    supporting_evidence: List[Evidence]
    contradicting_evidence: List[Evidence]
    neutral_evidence: List[Evidence]
    timestamp: str
    sources_analyzed: int

class RealFactChecker:
    def __init__(self):
        # initialize Gemini
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not found. Using basic analysis.")
        
        # news sources
        self.trusted_sources = {
            'thehindu.com': 0.9,
            'indianexpress.com': 0.85,
            'timesofindia.indiatimes.com': 0.8,
            'hindustantimes.com': 0.8,
            'business-standard.com': 0.85,
            
            'theprint.in': 0.85,
            'scroll.in': 0.8,
            'thewire.in': 0.8,
            'newslaundry.com': 0.85,
            
            'altnews.in': 0.95,
            'boomlive.in': 0.9,
            
            'nature.com': 0.95,
            'sciencemag.org': 0.95,
            'who.int': 0.9,
            'icmr.gov.in': 0.9
        }
    
    async def search_web(self, query: str, max_results: int = 10) -> List[dict]:
        try:
            results = []
            search_results = search(query, num_results=max_results, lang='en')
            
            for i, url in enumerate(search_results):
                if i >= max_results:
                    break
                
                try:
                    article_data = await self.extract_article_content(url)
                    if article_data:
                        results.append(article_data)
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    continue
            
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    async def extract_article_content(self, url: str) -> Optional[dict]:
        """Extract article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract domain for source identification
            domain = urlparse(url).netloc.lower()
            
            return {
                'title': article.title or 'No title',
                'url': url,
                'snippet': article.text[:300] + '...' if len(article.text) > 300 else article.text,
                'source': domain,
                'full_text': article.text,
                'publication_date': article.publish_date.isoformat() if article.publish_date else None
            }
        except Exception as e:
            print(f"Article extraction error for {url}: {str(e)}")
            return None
    
    def calculate_credibility_score(self, source: str) -> float:
        domain = source.lower()
        
        # exact matches first
        if domain in self.trusted_sources:
            return self.trusted_sources[domain]
        
        # partial matches
        for trusted_domain, score in self.trusted_sources.items():
            if trusted_domain in domain:
                return score
        
        # common reliable indicators
        if any(indicator in domain for indicator in ['.edu', '.gov', '.org']):
            return 0.7
        
        # potential unreliable indicators
        if any(indicator in domain for indicator in ['blog', 'wordpress', 'tumblr']):
            return 0.3
        
        # default score for unknown sources
        return 0.5
    
    async def analyze_with_gemini(self, claim: str, evidence_text: str) -> dict:
        if not self.model:
            return self.fallback_analysis(claim, evidence_text)
        
        try:
            prompt = f"""
            Analyze the following claim and evidence:
            
            CLAIM: {claim}
            
            EVIDENCE: {evidence_text}
            
            Please provide a JSON response with:
            1. "sentiment": "supporting", "contradicting", or "neutral"
            2. "relevance_score": float between 0-1
            3. "reasoning": explanation of your analysis
            4. "key_points": list of important points from the evidence
            
            Be objective and consider the credibility of sources.
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                response_text = response.text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # fallback parsing
                    return self.parse_gemini_response(response_text)
            except json.JSONDecodeError:
                return self.parse_gemini_response(response.text)
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return self.fallback_analysis(claim, evidence_text)
    
    def parse_gemini_response(self, text: str) -> dict:
        sentiment = 'neutral'
        relevance_score = 0.5
        reasoning = text
        
        text_lower = text.lower()
        if any(word in text_lower for word in ['supports', 'confirms', 'verifies', 'true']):
            sentiment = 'supporting'
            relevance_score = 0.7
        elif any(word in text_lower for word in ['contradicts', 'debunks', 'false', 'incorrect']):
            sentiment = 'contradicting'
            relevance_score = 0.7
        
        return {
            'sentiment': sentiment,
            'relevance_score': relevance_score,
            'reasoning': reasoning,
            'key_points': []
        }
    
    def fallback_analysis(self, claim: str, evidence_text: str) -> dict:
        claim_lower = claim.lower()
        evidence_lower = evidence_text.lower()
        
        # simple keyword matching
        contradicting_words = ['false', 'incorrect', 'wrong', 'debunked', 'myth', 'untrue', 'not true', 'misleading']
        supporting_words = ['true', 'correct', 'confirmed', 'verified', 'proven', 'accurate', 'factual']
        
        contradiction_score = sum(1 for word in contradicting_words if word in evidence_lower)
        supporting_score = sum(1 for word in supporting_words if word in evidence_lower)
        
        if contradiction_score > supporting_score:
            sentiment = 'contradicting'
        elif supporting_score > contradiction_score:
            sentiment = 'supporting'
        else:
            sentiment = 'neutral'
        
        # relevance based on keyword overlap
        claim_words = set(re.findall(r'\b[a-zA-Z]+\b', claim_lower))
        evidence_words = set(re.findall(r'\b[a-zA-Z]+\b', evidence_lower))
        
        if claim_words:
            relevance_score = len(claim_words.intersection(evidence_words)) / len(claim_words)
        else:
            relevance_score = 0.0
        
        return {
            'sentiment': sentiment,
            'relevance_score': relevance_score,
            'reasoning': f"Basic analysis: {sentiment} sentiment detected",
            'key_points': []
        }
    
    async def generate_comprehensive_summary(self, claim: str, all_evidence: List[Evidence]) -> str:
        if not self.model:
            return self.generate_basic_summary(claim, all_evidence)
        
        try:
            evidence_summary = "\n".join([
                f"- {e.source}: {e.sentiment} (relevance: {e.relevance_score:.2f})"
                for e in all_evidence[:10]  # cuz token limit
            ])
            
            prompt = f"""
            Provide a comprehensive fact-check summary for this claim:
            
            CLAIM: {claim}
            
            EVIDENCE SUMMARY:
            {evidence_summary}
            
            Please provide:
            1. Overall assessment of the claim's accuracy
            2. Key supporting and contradicting points
            3. Confidence level and reasoning
            4. Important caveats or nuances
            
            Be balanced and objective.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Summary generation error: {str(e)}")
            return self.generate_basic_summary(claim, all_evidence)
    
    def generate_basic_summary(self, claim: str, all_evidence: List[Evidence]) -> str:
        supporting = [e for e in all_evidence if e.sentiment == 'supporting']
        contradicting = [e for e in all_evidence if e.sentiment == 'contradicting']
        
        total_supporting_credibility = sum(e.credibility_score for e in supporting)
        total_contradicting_credibility = sum(e.credibility_score for e in contradicting)
        
        if total_supporting_credibility > total_contradicting_credibility:
            verdict = "appears to be TRUE"
        elif total_contradicting_credibility > total_supporting_credibility:
            verdict = "appears to be FALSE"
        else:
            verdict = "is UNCLEAR or MIXED"
        
        return f"Based on {len(all_evidence)} sources analyzed, the claim '{claim}' {verdict}. " \
               f"Found {len(supporting)} supporting sources and {len(contradicting)} contradicting sources."
    
    def calculate_accuracy_score(self, supporting: List[Evidence], contradicting: List[Evidence]) -> tuple:
        supporting_weight = sum(e.credibility_score * e.relevance_score for e in supporting)
        contradicting_weight = sum(e.credibility_score * e.relevance_score for e in contradicting)
        
        total_weight = supporting_weight + contradicting_weight
        
        if total_weight == 0:
            return 50.0, 'low'
        
        accuracy = (supporting_weight / total_weight) * 100
        
        total_evidence = len(supporting) + len(contradicting)
        avg_credibility = sum(e.credibility_score for e in supporting + contradicting) / max(total_evidence, 1)
        
        if total_evidence >= 5 and avg_credibility > 0.7:
            confidence = 'high'
        elif total_evidence >= 3 and avg_credibility > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return accuracy, confidence
    
    async def fact_check(self, claim: str, max_results: int = 5) -> FactCheckResult:
        """Main fact-checking function"""
        try:
            all_results = []
            
            # search for general information
            general_results = await self.search_web(claim, max_results)
            all_results.extend(general_results)
            
            fact_check_query = f"{claim} fact check verification"
            fact_check_results = await self.search_web(fact_check_query, max_results // 2)
            all_results.extend(fact_check_results)
            
            evidence_list = []
            
            for result in all_results:
                analysis = await self.analyze_with_gemini(claim, result['snippet'])
                
                credibility = self.calculate_credibility_score(result['source'])
                
                evidence = Evidence(
                    title=result['title'],
                    url=result['url'],
                    snippet=result['snippet'],
                    source=result['source'],
                    relevance_score=analysis['relevance_score'],
                    sentiment=analysis['sentiment'],
                    credibility_score=credibility,
                    publication_date=result.get('publication_date')
                )
                evidence_list.append(evidence)
            
            # categorize and sort evidence
            supporting = [e for e in evidence_list if e.sentiment == 'supporting']
            contradicting = [e for e in evidence_list if e.sentiment == 'contradicting']
            neutral = [e for e in evidence_list if e.sentiment == 'neutral']
            
            # sort by combined score (relevance × credibility)
            supporting.sort(key=lambda x: x.relevance_score * x.credibility_score, reverse=True)
            contradicting.sort(key=lambda x: x.relevance_score * x.credibility_score, reverse=True)
            neutral.sort(key=lambda x: x.relevance_score * x.credibility_score, reverse=True)
            
            # calculate accuracy and confidence
            accuracy, confidence = self.calculate_accuracy_score(supporting, contradicting)
            
            # generate summaries
            basic_summary = self.generate_basic_summary(claim, evidence_list)
            detailed_analysis = await self.generate_comprehensive_summary(claim, evidence_list)
            
            return FactCheckResult(
                claim=claim,
                accuracy_score=accuracy,
                confidence=confidence,
                summary=basic_summary,
                detailed_analysis=detailed_analysis,
                supporting_evidence=supporting[:max_results],
                contradicting_evidence=contradicting[:max_results],
                neutral_evidence=neutral[:max_results],
                timestamp=datetime.now().isoformat(),
                sources_analyzed=len(evidence_list)
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Fact-checking error: {str(e)}")

# Initialize fact checker
fact_checker = RealFactChecker()

@app.get("/")
async def root():
    return {
        "message": "Real Fact Checker API - Powered by web search and Gemini AI",
        "endpoints": {
            "fact_check": "/fact-check",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/fact-check", response_model=FactCheckResult)
async def check_fact(request: FactCheckRequest):
    """
    Fact-check a claim using real web search and AI analysis
    """
    try:
        if not request.claim.strip():
            raise HTTPException(status_code=400, detail="Claim cannot be empty")
        
        if len(request.claim) > 500:
            raise HTTPException(status_code=400, detail="Claim too long (max 500 characters)")
        
        result = await fact_checker.fact_check(request.claim, request.max_results)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    gemini_status = "available" if fact_checker.model else "not configured"
    return {
        "status": "healthy",
        "gemini_api": gemini_status,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn

    # env var check
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY=your_api_key_here")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
