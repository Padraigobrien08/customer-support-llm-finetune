import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Force dark mode for ChatGPT-like look
document.documentElement.classList.add("dark");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
