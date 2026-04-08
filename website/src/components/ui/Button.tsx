"use client";

import { forwardRef, ButtonHTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 font-medium transition-all duration-200 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan focus-visible:ring-offset-2 focus-visible:ring-offset-background",
  {
    variants: {
      variant: {
        primary: [
          "bg-cyan text-background hover:bg-cyan-dim",
          "font-mono uppercase tracking-wider",
          "shadow-[0_0_20px_rgba(0,255,255,0.3)]",
          "hover:shadow-[0_0_30px_rgba(0,255,255,0.5)]",
          "hover:-translate-y-0.5",
        ],
        secondary: [
          "bg-transparent text-cyan border border-cyan/30",
          "hover:border-cyan hover:bg-cyan/5",
          "font-mono uppercase tracking-wider",
        ],
        ghost: [
          "bg-transparent text-foreground-secondary",
          "hover:text-foreground hover:bg-white/5",
        ],
        danger: [
          "bg-error/10 text-error border border-error/30",
          "hover:bg-error/20 hover:border-error",
          "font-mono uppercase tracking-wider",
        ],
        amber: [
          "bg-amber text-background hover:bg-amber-dim",
          "font-mono uppercase tracking-wider",
          "shadow-[0_0_20px_rgba(255,176,0,0.3)]",
          "hover:shadow-[0_0_30px_rgba(255,176,0,0.5)]",
          "hover:-translate-y-0.5",
        ],
      },
      size: {
        sm: "px-4 py-2 text-xs rounded",
        md: "px-6 py-3 text-sm rounded-md",
        lg: "px-8 py-4 text-base rounded-md",
        icon: "p-2 rounded-md",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  isLoading?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, isLoading, children, disabled, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading && (
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

export { Button, buttonVariants };
