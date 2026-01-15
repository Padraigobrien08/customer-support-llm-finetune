import { MessageSquarePlus, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Thread } from "@/data/threads";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ThreadListProps {
  threads: Thread[];
  activeId: string;
  onSelect: (id: string) => void;
  onNewThread: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export function ThreadList({
  threads,
  activeId,
  onSelect,
  onNewThread,
  collapsed,
  onToggleCollapse
}: ThreadListProps) {
  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-slate-800/70 bg-slate-950/80 py-6 transition-all duration-200",
        collapsed ? "w-16 px-2" : "w-72 px-4"
      )}
    >
      <div className={cn("mb-6 flex items-center", collapsed ? "flex-col gap-3" : "justify-between")}>
        {collapsed ? (
          <div className="flex flex-col items-center gap-2">
            <Button variant="ghost" size="icon" onClick={onToggleCollapse}>
              <PanelLeftOpen className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onNewThread}>
              <MessageSquarePlus className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <>
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Conversations</h2>
              <p className="text-xs text-slate-500">Capability demos</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={onNewThread}>
                <MessageSquarePlus className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="icon" onClick={onToggleCollapse}>
                <PanelLeftClose className="h-4 w-4" />
              </Button>
            </div>
          </>
        )}
      </div>

      <div className="space-y-2 overflow-y-auto pb-4">
        {threads.map((thread) => (
          <button
            key={thread.id}
            onClick={() => onSelect(thread.id)}
            className={cn(
              "w-full rounded-lg border border-transparent text-left text-sm transition-colors",
              collapsed ? "px-2 py-2" : "px-3 py-2",
              activeId === thread.id
                ? "border-slate-700/80 bg-slate-900/70 text-slate-100"
                : "text-slate-400 hover:bg-slate-900/50"
            )}
          >
            {collapsed ? (
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-900/60 text-xs font-semibold uppercase text-slate-200">
                {thread.title.slice(0, 2)}
              </div>
            ) : (
              <>
                <div className="font-medium text-slate-200">{thread.title}</div>
                <div className="mt-1 line-clamp-1 text-xs text-slate-500">
                  {thread.messages[thread.messages.length - 1]?.content}
                </div>
              </>
            )}
          </button>
        ))}
      </div>
    </aside>
  );
}
