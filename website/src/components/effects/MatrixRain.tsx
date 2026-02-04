"use client";

import { useEffect, useRef, useCallback } from "react";

interface MatrixRainProps {
  className?: string;
  color?: string;
  fontSize?: number;
  speed?: number;
  opacity?: number;
}

export function MatrixRain({
  className = "",
  color = "#00ffff",
  fontSize = 14,
  speed = 33,
  opacity = 0.05,
}: MatrixRainProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<ReturnType<typeof setTimeout>>();

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, drops: number[], columns: number) => {
      // Semi-transparent black to create trail effect
      ctx.fillStyle = `rgba(10, 10, 10, 0.05)`;
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      // Characters to display
      const chars = "01COLLISION<>{}[]PROTOCOL";

      ctx.fillStyle = color;
      ctx.font = `${fontSize}px JetBrains Mono, monospace`;

      for (let i = 0; i < columns; i++) {
        const char = chars[Math.floor(Math.random() * chars.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;

        // Random brightness for depth effect
        const brightness = Math.random() * 0.5 + 0.5;
        ctx.fillStyle = `rgba(0, 255, 255, ${brightness * opacity * 2})`;
        ctx.fillText(char, x, y);

        // Reset drop randomly or when it goes off screen
        if (y > ctx.canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }

      return drops;
    },
    [color, fontSize, opacity]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resize();
    window.addEventListener("resize", resize);

    const columns = Math.floor(canvas.width / fontSize);
    let drops: number[] = Array(columns)
      .fill(0)
      .map(() => Math.random() * -100);

    const animate = () => {
      drops = draw(ctx, drops, columns);
      animationRef.current = setTimeout(() => {
        requestAnimationFrame(animate);
      }, speed);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resize);
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [draw, fontSize, speed]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ opacity }}
    />
  );
}
