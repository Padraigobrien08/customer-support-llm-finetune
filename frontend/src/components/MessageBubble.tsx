import { Copy, ThumbsDown, ThumbsUp, UserPlus, HelpCircle, ListOrdered, Info, Edit2 } from "lucide-react";
import { useState } from "react";
import { Message } from "@/data/threads";
import { cn, linkifyText, detectAction, assessResponseQuality } from "@/lib/utils";
import { MessageEditor } from "./MessageEditor";
import { QualityScore, type QualityMetrics } from "./QualityScore";

interface MessageBubbleProps {
  message: Message;
  onEdit?: (messageId: string, newContent: string) => void;
}

export function MessageBubble({ message, onEdit }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const detectedAction = !isUser ? detectAction(message.content) : null;
  const qualityRaw = !isUser ? assessResponseQuality(message.content) : null;
  
  // Convert quality to QualityMetrics format
  const quality: QualityMetrics | null = qualityRaw ? {
    score: qualityRaw.score,
    grade: qualityRaw.score >= 90 ? "A" : qualityRaw.score >= 80 ? "B" : qualityRaw.score >= 70 ? "C" : qualityRaw.score >= 60 ? "D" : "F",
    issues: qualityRaw.issues || [],
    strengths: qualityRaw.strengths || [],
    details: qualityRaw.details
  } : null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // no-op
    }
  };

  const handleEdit = (newContent: string) => {
    onEdit?.(message.id, newContent);
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <div className={cn("flex items-start gap-2", isUser ? "justify-end" : "justify-start")}>
        <MessageEditor
          initialContent={message.content}
          onSave={handleEdit}
          onCancel={() => setIsEditing(false)}
          isUser={isUser}
        />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "group flex items-start gap-2",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[85%] sm:max-w-[75%] md:max-w-[65%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          "min-w-0 w-full", // Ensure it can shrink
          isUser
            ? "bg-sky-400/20 text-sky-100"
            : "bg-slate-900/70 text-slate-100"
        )}
        title={message.timestamp ? `Sent ${message.timestamp}` : undefined}
      >
        <p className="whitespace-pre-wrap break-words" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>{linkifyText(message.content)}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {quality && <QualityScore quality={quality} compact />}
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
        {onEdit && (
          <button
            className="rounded-md border border-slate-800/70 bg-slate-900/70 p-1 text-slate-400 hover:text-slate-200"
            onClick={() => setIsEditing(true)}
            title="Edit message"
          >
            <Edit2 className="h-3.5 w-3.5" />
          </button>
        )}
        {copied && (
          <span className="text-[10px] text-slate-500">Copied</span>
        )}
      </div>
    </div>
  );
}
