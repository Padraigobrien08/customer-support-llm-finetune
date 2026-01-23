import { Copy, ThumbsDown, ThumbsUp, UserPlus, HelpCircle, ListOrdered, Info, CheckCircle2, AlertCircle } from "lucide-react";
import { useState } from "react";
import { Message } from "@/data/threads";
import { cn, linkifyText, detectAction, assessResponseQuality } from "@/lib/utils";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const detectedAction = !isUser ? detectAction(message.content) : null;
  const quality = !isUser ? assessResponseQuality(message.content) : null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // no-op
    }
  };

  return (
    <div
      className={cn(
        "group flex items-start gap-2",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[70%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-sky-400/20 text-sky-100"
            : "bg-slate-900/70 text-slate-100"
        )}
        title={message.timestamp ? `Sent ${message.timestamp}` : undefined}
      >
        <p className="whitespace-pre-wrap break-words">{linkifyText(message.content)}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {quality && quality.score < 70 && (
            <div
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                quality.score >= 50
                  ? "bg-yellow-500/20 text-yellow-300"
                  : "bg-red-500/20 text-red-300"
              )}
              title={quality.issues.join(", ")}
            >
              <AlertCircle className="h-3 w-3" />
              <span>Quality: {quality.score}%</span>
            </div>
          )}
          {quality && quality.score >= 70 && quality.strengths.length > 0 && (
            <div
              className="flex items-center gap-1.5 rounded-full bg-green-500/20 px-2.5 py-1 text-xs font-medium text-green-300"
              title={quality.strengths.join(", ")}
            >
              <CheckCircle2 className="h-3 w-3" />
              <span>Good quality</span>
            </div>
          )}
          {detectedAction && (
            <div
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                detectedAction.type === "escalation"
                  ? "bg-amber-500/20 text-amber-300"
                  : detectedAction.type === "clarification"
                  ? "bg-blue-500/20 text-blue-300"
                  : detectedAction.type === "instructions"
                  ? "bg-green-500/20 text-green-300"
                  : "bg-purple-500/20 text-purple-300"
              )}
            >
              {detectedAction.type === "escalation" ? (
                <UserPlus className="h-3 w-3" />
              ) : detectedAction.type === "clarification" ? (
                <HelpCircle className="h-3 w-3" />
              ) : detectedAction.type === "instructions" ? (
                <ListOrdered className="h-3 w-3" />
              ) : (
                <Info className="h-3 w-3" />
              )}
              <span>{detectedAction.label}</span>
            </div>
          )}
        </div>
      </div>
      <div
        className={cn(
          "flex flex-col gap-2 pt-2 opacity-0 transition-opacity group-hover:opacity-100",
          isUser ? "order-first" : "order-last"
        )}
      >
        <button
          className="rounded-md border border-slate-800/70 bg-slate-900/70 p-1 text-slate-400 hover:text-slate-200"
          onClick={handleCopy}
          title="Copy"
        >
          <Copy className="h-3.5 w-3.5" />
        </button>
        <button
          className="rounded-md border border-slate-800/70 bg-slate-900/70 p-1 text-slate-400 hover:text-slate-200"
          title="Thumbs up"
        >
          <ThumbsUp className="h-3.5 w-3.5" />
        </button>
        <button
          className="rounded-md border border-slate-800/70 bg-slate-900/70 p-1 text-slate-400 hover:text-slate-200"
          title="Thumbs down"
        >
          <ThumbsDown className="h-3.5 w-3.5" />
        </button>
        {copied && (
          <span className="text-[10px] text-slate-500">Copied</span>
        )}
      </div>
    </div>
  );
}
