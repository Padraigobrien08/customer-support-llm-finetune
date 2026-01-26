import { Check, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface MessageEditorProps {
  initialContent: string;
  onSave: (content: string) => void;
  onCancel: () => void;
  isUser?: boolean;
}

export function MessageEditor({ initialContent, onSave, onCancel, isUser = false }: MessageEditorProps) {
  const [content, setContent] = useState(initialContent);

  const handleSave = () => {
    if (content.trim()) {
      onSave(content.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      onCancel();
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleSave();
    }
  };

  return (
    <div className={cn("rounded-2xl px-4 py-3", isUser ? "bg-sky-400/20" : "bg-slate-900/70")}>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={handleKeyDown}
        className={cn(
          "w-full resize-none bg-transparent text-sm leading-relaxed outline-none",
          isUser ? "text-sky-100" : "text-slate-100"
        )}
        rows={Math.max(3, content.split("\n").length)}
        autoFocus
      />
      <div className="mt-2 flex items-center justify-end gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onCancel}
          className="h-7 text-xs text-slate-400 hover:text-slate-200"
        >
          <X className="h-3 w-3 mr-1" />
          Cancel
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          className="h-7 text-xs"
          disabled={!content.trim()}
        >
          <Check className="h-3 w-3 mr-1" />
          Save
        </Button>
      </div>
      <div className="mt-1 text-[10px] text-slate-500">
        Press Cmd/Ctrl+Enter to save, Esc to cancel
      </div>
    </div>
  );
}
