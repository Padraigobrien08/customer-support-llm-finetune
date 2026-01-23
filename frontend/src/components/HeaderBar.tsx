import { Cpu, Plug } from "lucide-react";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface HeaderBarProps {
  modelName: string;
  status: "connected" | "disconnected";
  environment?: string;
  settingsButton?: ReactNode;
}

export function HeaderBar({ modelName, status, environment, settingsButton }: HeaderBarProps) {
  const isConnected = status === "connected";

  return (
    <header className="flex items-center justify-between border-b border-slate-800/70 bg-slate-950/70 px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900/70">
          <Cpu className="h-4 w-4 text-slate-300" />
        </div>
        <div>
          <div className="text-sm font-semibold text-slate-100">
            Customer Support LLM
          </div>
          <div className="text-xs text-slate-500">{modelName}</div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {environment && (
          <span className="rounded-full bg-slate-900/70 px-3 py-1 text-xs text-slate-400">
            {environment}
          </span>
        )}
        {settingsButton}
        <span
          className={cn(
            "flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium",
            isConnected
              ? "bg-emerald-500/15 text-emerald-300"
              : "bg-rose-500/15 text-rose-300"
          )}
        >
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              isConnected ? "bg-emerald-400" : "bg-rose-400"
            )}
          />
          <Plug className="h-3 w-3" />
          {isConnected ? "Connected" : "Demo mode (not connected)"}
        </span>
      </div>
    </header>
  );
}
