import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ClaimForm from "./components/ClaimForm";
import FactResult from "./components/FactResult";
import ChatPage from "./components/ChatPage";

export default function App() {
  return (
    <Router>
      <Routes>
        {/* <Route path="/" element={<ClaimForm />} />
        <Route path="/result" element={<FactResult />} /> */}
        <Route path="/" element={<ChatPage/>}/>
      </Routes>
    </Router>
  );
}
