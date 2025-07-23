import React, { useState, useEffect, useRef } from "react";

export default function ChatPage() {
  // Add CSS animations
  React.useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      @keyframes slide-in {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      
      @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8); }
      }
      
      @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      
      @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
      }
      
      .animate-fade-in {
        animation: fade-in 0.5s ease-out;
      }
      
      .animate-slide-in {
        animation: slide-in 0.3s ease-out;
      }
      
      .animate-pulse-glow {
        animation: pulse-glow 2s ease-in-out infinite;
      }
      
      .animate-gradient {
        background-size: 300% 300%;
        animation: gradient-shift 3s ease infinite;
      }
      
      .animate-float {
        animation: float 3s ease-in-out infinite;
      }
      
      .glass-effect {
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
      }
      
      .input-glow:focus {
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
      }
      
      .sidebar-overlay {
        backdrop-filter: blur(5px);
        background: rgba(0, 0, 0, 0.3);
      }
      
      @media (max-width: 768px) {
        .sidebar-mobile {
          transform: translateX(-100%);
          transition: transform 0.3s ease-in-out;
        }
        
        .sidebar-mobile.open {
          transform: translateX(0);
        }
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const [darkMode, setDarkMode] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState({});
  const [currentSession, setCurrentSession] = useState(Date.now());
  const [input, setInput] = useState("");
  const [isMobile, setIsMobile] = useState(false);
  
  const chatEndRef = useRef(null);
  const chatRef = useRef(null);
  const inputRef = useRef(null);

  const messages = sessions[currentSession]?.messages || [];

  // Check if mobile on mount and window resize
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) {
        setSidebarOpen(false);
      }
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const updateSessionMessages = (id, newMessages) => {
    setSessions((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        messages: newMessages,
        title: prev[id]?.title || "Chat",
      },
    }));
  };

  const createNewSession = () => {
    const newId = Date.now();
    setSessions((prev) => ({
      ...prev,
      [newId]: { title: "New Chat", messages: [] },
    }));
    setCurrentSession(newId);
    if (isMobile) setSidebarOpen(false);
  };

  const deleteSession = (id) => {
    const { [id]: _, ...rest } = sessions;
    setSessions(rest);
    const keys = Object.keys(rest);
    const next = keys.length > 0 ? parseInt(keys[0]) : Date.now();
    setCurrentSession(next);
    if (!rest[next]) createNewSession();
  };

  const renameSession = (id) => {
    const newTitle = prompt("Enter new chat title:", sessions[id]?.title || "Chat");
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
      const backendUrl = import.meta.env.VITE_BACKEND_URL;

      const response = await fetch(`${backendUrl}/fact-check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: input, max_results: 5 }),
      });
      const data = await response.json();
      updateSessionMessages(currentSession, [
        ...newMessages,
        { role: "bot", content: data },
      ]);
    } catch {
      updateSessionMessages(currentSession, [
        ...newMessages,
        { role: "bot", content: "❌ Error fetching result." },
      ]);
    }
  };

  const getAccuracyColor = (score) => {
    if (score >= 80) return 'from-green-400 via-green-500 to-emerald-600';
    if (score >= 60) return 'from-yellow-400 via-yellow-500 to-orange-500';
    if (score >= 40) return 'from-orange-400 via-orange-500 to-red-500';
    return 'from-red-400 via-red-500 to-pink-600';
  };

  const getConfidenceIcon = (confidence) => {
    switch (confidence?.toLowerCase()) {
      case 'high': return '🔥';
      case 'medium': return '⚡';
      case 'low': return '💫';
      default: return '🤔';
    }
  };

  const renderFactResult = (fc) => {
    const accuracyScore = fc.accuracy_score || 0;
    const confidenceLevel = fc.confidence || 'unknown';
    
    return (
      <div className="space-y-6 animate-fade-in">
        {/* Header Card */}
        <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-blue-600 p-6 rounded-2xl text-white shadow-2xl animate-gradient">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center animate-float">
              🔍
            </div>
            <h3 className="font-bold text-xl">Fact-Check Analysis</h3>
          </div>
          <div className="text-sm opacity-90 leading-relaxed">
            <strong>Claim:</strong> {fc.claim}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Accuracy Score */}
          <div className={`bg-gradient-to-r ${getAccuracyColor(accuracyScore)} p-6 rounded-2xl text-white shadow-2xl transform hover:scale-105 transition-all duration-300 animate-float`}>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm opacity-90 font-medium">Accuracy Score</div>
                <div className="text-3xl font-bold">{accuracyScore.toFixed(1)}%</div>
              </div>
              <div className="text-3xl animate-pulse">📊</div>
            </div>
          </div>

          {/* Confidence Level */}
          <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 p-6 rounded-2xl text-white shadow-2xl transform hover:scale-105 transition-all duration-300 animate-float">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm opacity-90 font-medium">Confidence</div>
                <div className="text-xl font-bold capitalize">{confidenceLevel}</div>
              </div>
              <div className="text-3xl animate-pulse">{getConfidenceIcon(confidenceLevel)}</div>
            </div>
          </div>
        </div>

        {/* Summary Card */}
        <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-gray-800 dark:to-gray-700 p-6 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-600">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-sm">
              💡
            </div>
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 text-lg">Summary</h4>
          </div>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            {fc.summary}
          </p>
        </div>
         {fc.detailed_analysis && (
          <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-600">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-r from-gray-500 to-slate-600 rounded-full flex items-center justify-center text-white text-sm">
                📝
              </div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 text-lg">Detailed Analysis</h4>
            </div>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">
              {fc.detailed_analysis}
            </p>
          </div>
        )}
        {/* END: ADD THIS NEW SECTION */}


        {/* Evidence Sections */}
        <div className="space-y-6">
          {[
            { key: "supporting_evidence", title: "Supporting Evidence", icon: "✅", gradient: "from-green-400 via-green-500 to-emerald-600" },
            { key: "contradicting_evidence", title: "Contradicting Evidence", icon: "❌", gradient: "from-red-400 via-red-500 to-pink-600" },
            { key: "neutral_evidence", title: "Neutral Evidence", icon: "🟡", gradient: "from-yellow-400 via-yellow-500 to-orange-500" }
          ].map(({ key, title, icon, gradient }) => (
            <div key={key} className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-600 overflow-hidden">
              <div className={`bg-gradient-to-r ${gradient} p-4 animate-gradient`}>
                <div className="flex items-center gap-3 text-white">
                  <span className="text-xl">{icon}</span>
                  <h4 className="font-semibold text-lg">{title}</h4>
                  <span className="ml-auto bg-white/20 px-3 py-1 rounded-full text-sm font-medium">
                    {fc[key]?.length || 0} sources
                  </span>
                </div>
              </div>
              
              <div className="p-6">
                {fc[key]?.length ? (
                  <div className="space-y-4">
                    {fc[key].map((e, i) => (
                      <div key={i} className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl hover:shadow-lg transition-all duration-300 transform hover:scale-[1.02]">
                        <div className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                          {e.title}
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400 mb-3 leading-relaxed">
                          {e.snippet}
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-4 text-gray-500 dark:text-gray-400">
                            <span className="flex items-center gap-1">📰 {e.source}</span>
                            <span className="flex items-center gap-1">🎯 {e.relevance_score?.toFixed(2)}</span>
                            <span className="flex items-center gap-1">💭 {e.sentiment}</span>
                          </div>
                          <a
                            href={e.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-full hover:shadow-lg transition-all duration-300 transform hover:scale-105 font-medium"
                          >
                            View Source
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                    <div className="text-5xl mb-3 animate-pulse">🔍</div>
                    <div className="text-lg">No evidence found</div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const copyToClipboard = (text) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        alert("Copied to clipboard!");
      })
      .catch(() => {
        alert("Failed to copy.");
      });
  };

  const exportChatToPDF = () => {
    const sessionTitle = sessions[currentSession]?.title || "Chat";
    const now = new Date().toLocaleString();
    
    const printWindow = window.open('', '_blank');
    
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>${sessionTitle} - Fact-Check Report</title>
        <style>
          body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            padding: 20px; 
            max-width: 800px; 
            margin: 0 auto;
            line-height: 1.6;
          }
          h2 { text-align: center; color: #333; margin-bottom: 10px; }
          .header { text-align: center; font-size: 0.95em; color: #666; margin-bottom: 30px; }
          .message { margin-bottom: 20px; }
          .user-message { 
            background: #e3f2fd; 
            padding: 12px 16px; 
            border-radius: 8px; 
            margin-left: 20%;
          }
          .bot-message { 
            background: #f5f5f5; 
            padding: 12px 16px; 
            border-radius: 8px; 
            margin-right: 20%;
          }
          .message-label { font-weight: bold; margin-bottom: 8px; }
          .fact-check { padding: 4px 0; }
          .fact-check strong { color: #1976d2; }
          hr { margin: 20px 0; border: none; border-top: 1px solid #ddd; }
          @media print {
            body { margin: 0; padding: 15px; }
            .no-print { display: none !important; }
          }
        </style>
      </head>
      <body>
        <h2>🧾 Fact-Check Report</h2>
        <div class="header">
          <strong>Title:</strong> ${sessionTitle}<br/>
          <strong>Date:</strong> ${now}
        </div>
        <hr />
        ${messages
          .map((msg) => {
            if (msg.role === "user") {
              return `
                <div class="message">
                  <div class="message-label">🧍 You:</div>
                  <div class="user-message">${msg.content}</div>
                </div>
              `;
            } else if (msg.role === "bot" && typeof msg.content === "object") {
              const fc = msg.content;
              return `
                <div class="message">
                  <div class="message-label">🔍 Fact-Check:</div>
                  <div class="bot-message">
                    <div class="fact-check"><strong>Claim:</strong> ${fc.claim}</div>
                    <div class="fact-check"><strong>Accuracy Score:</strong> ${fc.accuracy_score?.toFixed(1) || 'N/A'}%</div>
                    <div class="fact-check"><strong>Confidence:</strong> ${fc.confidence || 'N/A'}</div>
                    <div class="fact-check"><strong>Summary:</strong> ${fc.summary || 'N/A'}</div>
                  </div>
                </div>
              `;
            } else {
              return `
                <div class="message">
                  <div class="message-label">🔍 Fact-Check:</div>
                  <div class="bot-message">${msg.content}</div>
                </div>
              `;
            }
          })
          .join("")}
        <div style="margin-top: 40px; text-align: center; color: #666; font-size: 0.9em;">
          Generated on ${now}
        </div>
      </body>
      </html>
    `;
    
    printWindow.document.write(htmlContent);
    printWindow.document.close();
    
    printWindow.onload = () => {
      printWindow.print();
      printWindow.close();
    };
  };

  return (
    <div className={`flex min-h-screen relative ${
      darkMode ? "bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900" : "bg-gradient-to-br from-blue-50 via-white to-purple-50"
    }`}>
      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`${
        isMobile ? 'fixed left-0 top-0 h-full z-50 sidebar-mobile' : 'relative'
      } ${sidebarOpen ? 'open' : ''} ${
        isMobile ? 'w-80' : sidebarOpen ? 'w-80' : 'w-0'
      } transition-all duration-300 ease-in-out overflow-hidden border-r ${
        darkMode ? "bg-gradient-to-b from-gray-800 via-gray-900 to-black text-white border-gray-700" : "bg-gradient-to-b from-white via-gray-50 to-white text-black border-gray-200"
      } shadow-2xl`}>
        <div className="p-6 w-80">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Chat History
            </h2>
            {isMobile && (
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          
          <button
            onClick={createNewSession}
            className="bg-gradient-to-r from-blue-500 via-purple-600 to-pink-600 hover:from-blue-600 hover:via-purple-700 hover:to-pink-700 text-white w-full py-4 rounded-2xl mb-6 transition-all duration-300 transform hover:scale-105 shadow-xl font-semibold text-lg animate-gradient"
          >
            ✨ New Chat
          </button>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {Object.entries(sessions).map(([id, sess]) => (
              <div key={id} className="relative group">
                <button
                  onClick={() => {
                    setCurrentSession(parseInt(id));
                    if (isMobile) setSidebarOpen(false);
                  }}
                  className={`w-full text-left p-4 rounded-xl transition-all duration-300 transform hover:scale-105 ${
                    parseInt(id) === currentSession
                      ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl"
                      : "hover:bg-gray-100 dark:hover:bg-gray-700 hover:shadow-lg"
                  }`}
                >
                  <div className="truncate font-medium">
                    {sess.title || `Chat ${new Date(parseInt(id)).toLocaleDateString()}`}
                  </div>
                  <div className="text-xs opacity-70 mt-1">
                    {sess.messages?.length || 0} messages
                  </div>
                </button>
                
                <div className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 flex space-x-1 transition-opacity">
                  <button
                    onClick={() => renameSession(parseInt(id))}
                    className="text-sm hover:text-blue-600 transition-colors p-1 rounded"
                    title="Rename"
                  >
                    ✏️
                  </button>
                  <button
                    onClick={() => deleteSession(parseInt(id))}
                    className="text-sm hover:text-red-600 transition-colors p-1 rounded"
                    title="Delete"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className={`flex justify-between items-center p-4 border-b backdrop-blur-sm ${
          darkMode ? "bg-gray-800/50 border-gray-700" : "bg-white/50 border-gray-200"
        } shadow-lg`}>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`p-3 rounded-xl transition-all duration-300 transform hover:scale-105 ${
                darkMode 
                  ? "bg-gradient-to-r from-gray-700 to-gray-600 hover:from-gray-600 hover:to-gray-500 text-white shadow-lg" 
                  : "bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 shadow-lg"
              }`}
              title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <button
              onClick={exportChatToPDF}
              className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white px-6 py-3 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg font-semibold"
            >
              📄 Export
            </button>
          </div>
          
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`px-6 py-3 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg font-semibold ${
              darkMode 
                ? "bg-gradient-to-r from-yellow-400 to-orange-500 hover:from-yellow-500 hover:to-orange-600 text-white" 
                : "bg-gradient-to-r from-slate-800 to-gray-900 hover:from-slate-900 hover:to-black text-white"
            }`}
          >
            {darkMode ? "🌞 Light" : "🌙 Dark"}
          </button>
        </div>

        {/* Chat Messages */}
        <div className="flex-grow overflow-y-auto p-6 space-y-6" ref={chatRef}>
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-6xl mb-4 animate-float">🔍</div>
                <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Ready to Fact-Check
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-lg">
                  Ask me anything and I'll verify the facts for you!
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`relative group rounded-2xl p-6 max-w-[85%] whitespace-pre-wrap transition-all duration-300 hover:shadow-xl ${
                  msg.role === "user"
                    ? "ml-auto bg-gradient-to-r from-blue-500 via-purple-600 to-pink-600 text-white shadow-xl animate-gradient"
                    : "mr-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-black dark:text-white shadow-xl"
                }`}
              >
                <button
                  onClick={() =>
                    copyToClipboard(
                      typeof msg.content === "object"
                        ? `Claim: ${msg.content.claim}\nSummary: ${msg.content.summary}`
                        : msg.content
                    )
                  }
                  className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 bg-black/20 hover:bg-black/40 text-white p-2 rounded-full transition-all duration-300 transform hover:scale-110"
                  title="Copy to clipboard"
                >
                  📋
                </button>
                
                {msg.role === "bot" && typeof msg.content === "object"
                  ? renderFactResult(msg.content)
                  : msg.content}
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area - ChatGPT Style */}
        <div className="p-6">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmit} className="relative">
              <div className={`relative rounded-2xl shadow-2xl ${
                darkMode 
                  ? "bg-gray-800 border border-gray-600" 
                  : "bg-white border border-gray-200"
              }`}>
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                  placeholder="Ask a fact-checking question..."
                  rows={1}
                  className={`w-full p-6 pr-16 resize-none rounded-2xl focus:outline-none input-glow transition-all duration-300 ${
                    darkMode 
                      ? "bg-gray-800 text-white placeholder-gray-400" 
                      : "bg-white text-black placeholder-gray-500"
                  }`}
                  style={{ 
                    minHeight: '60px',
                    maxHeight: '200px',
                    overflowY: input.split('\n').length > 3 ? 'auto' : 'hidden'
                  }}
                />
                
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 p-3 rounded-xl transition-all duration-300 ${
                    input.trim()
                      ? "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg transform hover:scale-105 cursor-pointer"
                      : "bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed"
                  }`}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
              
              <div className="text-center mt-3 text-xs text-gray-500 dark:text-gray-400">
                Press Enter to send, Shift+Enter for new line
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}