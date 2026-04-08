"use client";

import Link from "next/link";
import { Zap, Github, Twitter, MessageCircle } from "lucide-react";

const footerLinks = {
  product: [
    { label: "Pool Stats", href: "/pool" },
    { label: "Download", href: "/download" },
    { label: "Dashboard", href: "/dashboard" },
    { label: "Pricing", href: "/#pricing" },
  ],
  resources: [
    { label: "Documentation", href: "/docs" },
    { label: "Getting Started", href: "/docs/getting-started" },
    { label: "FAQ", href: "/docs/faq" },
    { label: "API Reference", href: "/docs/api" },
  ],
  company: [
    { label: "About", href: "/about" },
    { label: "Blog", href: "/blog" },
    { label: "Terms", href: "/terms" },
    { label: "Privacy", href: "/privacy" },
  ],
};

const socialLinks = [
  { label: "GitHub", href: "https://github.com/hevnsnt/theCollider", icon: Github },
  { label: "Twitter", href: "https://twitter.com", icon: Twitter },
  { label: "Discord", href: "https://discord.gg", icon: MessageCircle },
];

export function Footer() {
  return (
    <footer className="bg-background-secondary border-t border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-8 lg:gap-12">
          {/* Brand Column */}
          <div className="col-span-2 lg:col-span-1">
            <Link href="/" className="flex items-center gap-2 mb-4">
              <Zap className="w-6 h-6 text-cyan" />
              <span className="font-mono text-sm font-bold">
                <span className="text-foreground">COLLISION</span>
                <span className="text-cyan">PROTOCOL</span>
              </span>
            </Link>
            <p className="text-sm text-foreground-secondary mb-6 max-w-xs">
              Distributed GPU computing for Bitcoin puzzle solving. Join thousands of GPUs working together.
            </p>
            <div className="flex gap-4">
              {socialLinks.map((link) => (
                <Link
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-foreground-muted hover:text-cyan transition-colors"
                  aria-label={link.label}
                >
                  <link.icon className="w-5 h-5" />
                </Link>
              ))}
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 className="font-mono text-xs uppercase tracking-wider text-foreground-muted mb-4">
              Product
            </h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-foreground-secondary hover:text-cyan transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources Links */}
          <div>
            <h3 className="font-mono text-xs uppercase tracking-wider text-foreground-muted mb-4">
              Resources
            </h3>
            <ul className="space-y-3">
              {footerLinks.resources.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-foreground-secondary hover:text-cyan transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3 className="font-mono text-xs uppercase tracking-wider text-foreground-muted mb-4">
              Company
            </h3>
            <ul className="space-y-3">
              {footerLinks.company.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-foreground-secondary hover:text-cyan transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Status */}
          <div>
            <h3 className="font-mono text-xs uppercase tracking-wider text-foreground-muted mb-4">
              Pool Status
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="status-online" />
                <span className="text-sm text-foreground-secondary">Pool Online</span>
              </div>
              <p className="text-xs text-foreground-muted font-mono">
                Uptime: 99.9%
              </p>
              <Link
                href="/status"
                className="text-sm text-cyan hover:text-cyan-dim transition-colors"
              >
                View Status Page &rarr;
              </Link>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs text-foreground-muted">
            &copy; {new Date().getFullYear()} Collision Protocol. All rights reserved.
          </p>
          <p className="text-xs text-foreground-muted font-mono">
            Built for the Bitcoin Puzzle Challenge
          </p>
        </div>
      </div>
    </footer>
  );
}
