import React from "react";
import { useNavigate } from "react-router-dom";

export default function FactResult() {
  const navigate = useNavigate();
  const data = JSON.parse(localStorage.getItem("factResult"));

  if (!data) {
    return (
      <div className="p-4 text-center">
        <p>No result found. Please go back and submit a claim.</p>
        <button onClick={() => navigate("/")} className="mt-2 underline text-blue-600">
          Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-semibold text-blue-700 mb-2">
          Claim: <span className="text-black">{data.claim}</span>
        </h2>
        <p className="text-gray-700 mb-4">
          <strong>Accuracy Score:</strong> {data.accuracy_score.toFixed(1)}%<br />
          <strong>Confidence:</strong> {data.confidence}
        </p>
        <p className="mb-4">
          <strong>Summary:</strong> {data.summary}
        </p>

        <section className="mt-6">
          <h3 className="text-lg font-bold text-green-600">✅ Supporting Evidence</h3>
          {data.supporting_evidence.map((e, i) => (
            <EvidenceCard key={i} {...e} />
          ))}
        </section>

        <section className="mt-6">
          <h3 className="text-lg font-bold text-red-600">❌ Contradicting Evidence</h3>
          {data.contradicting_evidence.map((e, i) => (
            <EvidenceCard key={i} {...e} />
          ))}
        </section>

        <section className="mt-6">
          <h3 className="text-lg font-bold text-gray-600">🟡 Neutral Evidence</h3>
          {data.neutral_evidence.map((e, i) => (
            <EvidenceCard key={i} {...e} />
          ))}
        </section>
      </div>
    </div>
  );
}

function EvidenceCard({ title, url, snippet, source, relevance_score, sentiment }) {
  return (
    <div className="border border-gray-200 rounded p-4 my-2 bg-gray-50">
      <h4 className="font-semibold text-blue-800">{title}</h4>
      <p className="text-sm text-gray-700 mb-2">{snippet}</p>
      <p className="text-sm text-gray-500">Source: {source}</p>
      <p className="text-sm">Relevance: {relevance_score.toFixed(2)} | Sentiment: {sentiment}</p>
      <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline text-sm">
        View Source
      </a>
    </div>
  );
}
