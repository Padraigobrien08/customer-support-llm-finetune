import * as React from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "ghost";
  size?: "default" | "sm" | "icon";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/50 disabled:pointer-events-none disabled:opacity-50",
          variant === "default" &&
            "bg-sky-400 text-slate-950 hover:bg-sky-300",
          variant === "secondary" &&
            "bg-slate-800/80 text-slate-100 hover:bg-slate-700/80",
          variant === "ghost" &&
            "bg-transparent text-slate-200 hover:bg-slate-800/70",
          size === "default" && "h-10 px-4 py-2",
          size === "sm" && "h-9 px-3",
          size === "icon" && "h-9 w-9",
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button };
