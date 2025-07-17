// ✅ ChatPage.jsx with Rename/Delete Chats + Dark Mode Toggle (Top-Right) — FIXED undefined.map crash

import React, { useState, useEffect, useRef } from "react";

export default function ChatPage() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem("darkMode");
    return saved ? JSON.parse(saved) : false;
  });

  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem("chatSessions");
    return saved ? JSON.parse(saved) : {};
  });

  const [currentSession, setCurrentSession] = useState(() => {
    const keys = Object.keys(sessions);
    return keys.length > 0 ? parseInt(keys[0]) : Date.now();
  });

  const [input, setInput] = useState("");
  const chatEndRef = useRef(null);

  const messages = sessions[currentSession]?.messages || [];

  useEffect(() => {
    localStorage.setItem("chatSessions", JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(darkMode));
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const updateSessionMessages = (id, newMessages) => {
    setSessions((prev) => ({
      ...prev,
      [id]: {
        ...(prev[id] || { title: "New Chat" }),
        messages: newMessages,
      },
    }));
  };

  const createNewSession = () => {
    const newId = Date.now();
    setSessions((prev) => ({ ...prev, [newId]: { title: "New Chat", messages: [] } }));
    setCurrentSession(newId);
  };

  const deleteSession = (id) => {
    const { [id]: _, ...rest } = sessions;
    setSessions(rest);
    const keys = Object.keys(rest);
    const next = keys.length > 0 ? parseInt(keys[0]) : Date.now();
    setCurrentSession(next);
    if (!rest[next]) {
      createNewSession();
    }
  };

  const renameSession = (id) => {
    const newTitle = prompt("Enter new chat title:", sessions[id]?.title || "");
    if (newTitle !== null && newTitle.trim()) {
      setSessions((prev) => ({
        ...prev,
        [id]: { ...prev[id], title: newTitle.trim() },
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessages = [...messages, { role: "user", content: input }];
    updateSessionMessages(currentSession, newMessages);
    setInput("");

    try {
      const response = await fetch("/fact-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: input, max_results: 5 }),
      });
      const data = await response.json();
      updateSessionMessages(currentSession, [...newMessages, { role: "bot", content: data }]);
    } catch {
      updateSessionMessages(currentSession, [...newMessages, { role: "bot", content: "❌ Error fetching result." }]);
    }
  };

  const renderFactResult = (data) => {
    if (typeof data !== "object") return data;
    return (
      <div className="space-y-4">
        <div>
          <strong>Claim:</strong> {data.claim}
          <br />
          <strong>Accuracy Score:</strong> {data.accuracy_score.toFixed(1)}%
          <br />
          <strong>Confidence:</strong> {data.confidence}
          <br />
          <strong>Summary:</strong> {data.summary}
        </div>

        {["supporting_evidence", "contradicting_evidence", "neutral_evidence"].map((type, idx) => {
          const label = {
            supporting_evidence: "✅ Supporting Evidence",
            contradicting_evidence: "❌ Contradicting Evidence",
            neutral_evidence: "🟡 Neutral Evidence",
          }[type];
          const color = {
            supporting_evidence: "text-green-600",
            contradicting_evidence: "text-red-600",
            neutral_evidence: "text-gray-600",
          }[type];
          return (
            <section key={idx}>
              <h3 className={`text-lg font-bold ${color}`}>{label}</h3>
              {Array.isArray(data[type]) && data[type].map((e, i) => (
                <div key={i} className="border border-gray-200 rounded p-4 my-2 bg-gray-50 dark:bg-gray-800">
                  <h4 className="font-semibold text-blue-800 dark:text-blue-400">{e.title}</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">{e.snippet}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Source: {e.source}</p>
                  <p className="text-sm dark:text-gray-300">
                    Relevance: {e.relevance_score.toFixed(2)} | Sentiment: {e.sentiment}
                  </p>
                  <a href={e.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-300 underline text-sm">
                    View Source
                  </a>
                </div>
              ))}
            </section>
          );
        })}
      </div>
    );
  };

  return (
    <div className={`flex min-h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-100'} transition-colors`}>
      <div className={`w-64 p-4 space-y-4 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-300'} border-r`}>
        <h2 className={`text-xl font-bold ${darkMode ? 'text-blue-300' : 'text-blue-700'}`}>Chat History</h2>
        <button onClick={createNewSession} className="bg-blue-600 text-white py-1 px-3 rounded-md hover:bg-blue-700 w-full">
          + New Chat
        </button>
        <ul className="space-y-2">
          {Object.entries(sessions).map(([id, sess]) => (
            <li key={id} className="relative group">
              <button
                className={`text-left w-full p-2 rounded-md ${parseInt(id) === currentSession ? (darkMode ? 'bg-blue-700 text-white' : 'bg-blue-100 text-blue-800') : 'hover:bg-gray-200 dark:hover:bg-gray-700'}`}
                onClick={() => setCurrentSession(parseInt(id))}
              >
                {sess.title || `Chat ${new Date(parseInt(id)).toLocaleString()}`}
              </button>
              <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 flex space-x-1">
                <button onClick={() => renameSession(parseInt(id))} className="text-sm p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded">✏️</button>
                <button onClick={() => deleteSession(parseInt(id))} className="text-sm p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded">🗑️</button>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex-1 flex flex-col">
        <div className="flex justify-end p-4 border-b dark:border-gray-700">
          <button
            onClick={() => setDarkMode((dm) => !dm)}
            className="px-4 py-1 rounded-md text-sm font-medium bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:opacity-80"
          >
            {darkMode ? '🌞 Light Mode' : '🌙 Dark Mode'}
          </button>
        </div>

        <div className="flex-grow overflow-y-auto p-6 space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={`rounded-lg p-3 max-w-[90%] whitespace-pre-wrap ${
              msg.role === 'user'
                ? 'ml-auto bg-blue-600 text-white'
                : `mr-auto bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-black dark:text-white`
            }`}>
              {msg.role === 'bot' && typeof msg.content === 'object' ? renderFactResult(msg.content) : msg.content}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleSubmit} className={`flex border-t ${darkMode ? 'border-gray-700' : 'border-gray-300'}`}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-grow p-4 border-none focus:outline-none bg-white dark:bg-gray-700 text-black dark:text-white"
            placeholder="Ask a fact-checking question..."
          />
          <button type="submit" className="px-6 text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50" disabled={!input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}