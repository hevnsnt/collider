"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, Zap, Github, Download } from "lucide-react";
import { Button } from "@/components/ui";
import { cn } from "@/lib/utils";

const navLinks = [
  { href: "/pool", label: "Pool Stats" },
  { href: "/download", label: "Download" },
  { href: "/docs", label: "Docs" },
];

export function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        isScrolled
          ? "bg-background/80 backdrop-blur-md border-b border-border"
          : "bg-transparent"
      )}
    >
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <div className="relative">
              <Zap className="w-8 h-8 text-cyan transition-transform group-hover:scale-110" />
              <div className="absolute inset-0 blur-md bg-cyan/30 group-hover:bg-cyan/50 transition-colors" />
            </div>
            <span className="font-mono text-lg font-bold tracking-tight">
              <span className="text-foreground">COLLISION</span>
              <span className="text-cyan">PROTOCOL</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="font-mono text-sm text-foreground-secondary hover:text-cyan transition-colors relative group"
              >
                {link.label}
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-cyan transition-all group-hover:w-full" />
              </Link>
            ))}
          </div>

          {/* Desktop Actions */}
          <div className="hidden lg:flex items-center gap-4">
            <Link
              href="https://github.com/hevnsnt/theCollider"
              target="_blank"
              rel="noopener noreferrer"
              className="text-foreground-secondary hover:text-foreground transition-colors"
            >
              <Github className="w-5 h-5" />
            </Link>
            <Link href="/download">
              <Button variant="secondary" size="sm">
                <Download className="w-4 h-4" />
                Download
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button size="sm">Launch App</Button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="lg:hidden p-2 text-foreground-secondary hover:text-foreground"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden bg-background/95 backdrop-blur-md border-b border-border"
          >
            <div className="px-4 py-6 space-y-4">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="block font-mono text-foreground-secondary hover:text-cyan transition-colors py-2"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {link.label}
                </Link>
              ))}
              <div className="pt-4 flex flex-col gap-3">
                <Link href="/download" onClick={() => setIsMobileMenuOpen(false)}>
                  <Button variant="secondary" className="w-full">
                    <Download className="w-4 h-4" />
                    Download Client
                  </Button>
                </Link>
                <Link href="/dashboard" onClick={() => setIsMobileMenuOpen(false)}>
                  <Button className="w-full">Launch App</Button>
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
