import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function ClaimForm() {
  const [claim, setClaim] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!claim.trim()) return;

    setLoading(true);

    try {
      const backendUrl = import.meta.env.VITE_BACKEND_URL;

      const response = await fetch(`${backendUrl}/fact-check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim, max_results: 5 }),
      });

      const data = await response.json();
      localStorage.setItem("factResult", JSON.stringify(data));
      navigate("/result");
    } catch (err) {
      alert("Error fetching fact check result");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-100 to-white px-4">
      <form
        onSubmit={handleSubmit}
        className="bg-white shadow-xl rounded-xl p-8 max-w-xl w-full"
      >
        <h1 className="text-3xl font-bold mb-4 text-center text-blue-700">
          Fact Checker
        </h1>
        <textarea
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          rows={5}
          className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
          placeholder="Enter a claim to verify its accuracy"
          required
        />
        <button
          type="submit"
          className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition"
          disabled={loading}
        >
          {loading ? "Verifying..." : "Check Fact"}
        </button>
      </form>
    </div>
  );
}
