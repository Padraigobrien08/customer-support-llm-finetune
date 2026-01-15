import { Copy, ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { Message } from "@/data/threads";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

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
      >
        <p>{message.content}</p>
        <span className="mt-2 block text-[10px] uppercase tracking-wide text-slate-500">
          {message.timestamp}
        </span>
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
