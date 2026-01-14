import { RefreshCcw, SendHorizontal, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface PromptComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onGeneratePrompt: () => void;
  isGenerating?: boolean;
}

export function PromptComposer({
  value,
  onChange,
  onSend,
  onGeneratePrompt,
  isGenerating
}: PromptComposerProps) {
  return (
    <div className="border-t border-slate-800 bg-slate-950 p-4">
      <div className="flex items-center gap-3">
        <Button variant="secondary" size="sm" onClick={onGeneratePrompt}>
          {isGenerating ? (
            <RefreshCcw className="h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="h-4 w-4" />
          )}
          Example prompt
        </Button>
        <div className="flex flex-1 items-center gap-2">
          <Input
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Ask the model anything about support workflows..."
          />
          <Button onClick={onSend} disabled={!value.trim()}>
            <SendHorizontal className="h-4 w-4" />
            Send
          </Button>
        </div>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Model backend connection is not wired yet. This UI is ready for integration.
      </p>
    </div>
  );
}
