import React from "react";
import { Outlet } from "react-router-dom";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export const MainLayout: React.FC = () => {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="main-content-area">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};
