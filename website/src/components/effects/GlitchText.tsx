"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface GlitchTextProps {
  text: string;
  className?: string;
  as?: "h1" | "h2" | "h3" | "h4" | "span" | "p";
  glitchOnHover?: boolean;
  continuous?: boolean;
}

export function GlitchText({
  text,
  className = "",
  as: Component = "span",
  glitchOnHover = false,
  continuous = false,
}: GlitchTextProps) {
  const [isGlitching, setIsGlitching] = useState(continuous);

  useEffect(() => {
    if (!continuous) return;

    const interval = setInterval(() => {
      setIsGlitching(true);
      setTimeout(() => setIsGlitching(false), 200);
    }, 3000 + Math.random() * 2000);

    return () => clearInterval(interval);
  }, [continuous]);

  return (
    <Component
      className={cn(
        "relative inline-block",
        glitchOnHover && "group",
        className
      )}
      onMouseEnter={() => glitchOnHover && setIsGlitching(true)}
      onMouseLeave={() => glitchOnHover && setIsGlitching(false)}
      data-text={text}
    >
      <span className="relative z-10">{text}</span>
      {isGlitching && (
        <>
          <span
            className="absolute top-0 left-0 w-full h-full text-cyan opacity-80 animate-glitch"
            style={{
              clipPath: "polygon(0 0, 100% 0, 100% 45%, 0 45%)",
              transform: "translate(-2px, 0)",
            }}
            aria-hidden="true"
          >
            {text}
          </span>
          <span
            className="absolute top-0 left-0 w-full h-full text-error opacity-80 animate-glitch"
            style={{
              clipPath: "polygon(0 55%, 100% 55%, 100% 100%, 0 100%)",
              transform: "translate(2px, 0)",
              animationDelay: "0.05s",
            }}
            aria-hidden="true"
          >
            {text}
          </span>
        </>
      )}
    </Component>
  );
}
