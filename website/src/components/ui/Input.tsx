"use client";

import { forwardRef, InputHTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const inputVariants = cva(
  [
    "w-full bg-background border border-border rounded-md font-mono text-sm text-foreground",
    "placeholder:text-foreground-muted",
    "transition-all duration-200",
    "focus:outline-none focus:border-cyan focus:ring-1 focus:ring-cyan/30",
    "focus:shadow-[0_0_0_3px_rgba(0,255,255,0.1),inset_0_0_20px_rgba(0,255,255,0.05)]",
    "disabled:opacity-50 disabled:cursor-not-allowed",
  ],
  {
    variants: {
      inputSize: {
        sm: "px-3 py-2 text-xs",
        md: "px-4 py-3 text-sm",
        lg: "px-5 py-4 text-base",
      },
    },
    defaultVariants: {
      inputSize: "md",
    },
  }
);

export interface InputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "size">,
    VariantProps<typeof inputVariants> {
  icon?: React.ReactNode;
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, inputSize, icon, error, ...props }, ref) => {
    return (
      <div className="relative">
        {icon && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-foreground-muted">
            {icon}
          </div>
        )}
        <input
          type={type}
          className={cn(
            inputVariants({ inputSize }),
            icon && "pl-10",
            error && "border-error focus:border-error focus:ring-error/30",
            className
          )}
          ref={ref}
          {...props}
        />
        {error && (
          <p className="mt-1.5 text-xs text-error font-mono">{error}</p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";

export { Input, inputVariants };
