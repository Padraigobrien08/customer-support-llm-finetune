import { MessageSquarePlus } from "lucide-react";
import { Thread } from "@/data/threads";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ThreadListProps {
  threads: Thread[];
  activeId: string;
  onSelect: (id: string) => void;
  onNewThread: () => void;
}

export function ThreadList({ threads, activeId, onSelect, onNewThread }: ThreadListProps) {
  return (
    <aside className="flex h-full w-72 flex-col border-r border-slate-800/70 bg-slate-950/80 px-4 py-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Conversations</h2>
          <p className="text-xs text-slate-500">Capability demos</p>
        </div>
        <Button variant="ghost" size="icon" onClick={onNewThread}>
          <MessageSquarePlus className="h-4 w-4" />
        </Button>
      </div>
      <div className="space-y-2 overflow-y-auto pb-4">
        {threads.map((thread) => (
          <button
            key={thread.id}
            onClick={() => onSelect(thread.id)}
            className={cn(
              "w-full rounded-lg border border-transparent px-3 py-2 text-left text-sm transition-colors",
              activeId === thread.id
                ? "border-slate-700/80 bg-slate-900/70 text-slate-100"
                : "text-slate-400 hover:bg-slate-900/50"
            )}
          >
            <div className="font-medium text-slate-200">{thread.title}</div>
            <div className="mt-1 line-clamp-1 text-xs text-slate-500">
              {thread.messages[thread.messages.length - 1]?.content}
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}
