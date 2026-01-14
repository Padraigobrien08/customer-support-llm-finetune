import { Message } from "@/data/threads";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}> 
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
    </div>
  );
}
