import { Thread } from "@/data/threads";
import { MessageBubble } from "@/components/MessageBubble";

interface ChatWindowProps {
  thread: Thread;
  isTyping?: boolean;
  onClearThread: () => void;
  onRenameThread: (title: string) => void;
}

export function ChatWindow({
  thread,
  isTyping = false,
  onClearThread,
  onRenameThread
}: ChatWindowProps) {
  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-6 py-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">{thread.title}</h1>
          <p className="text-xs text-slate-500">
            Demo thread showing model capability
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            className="rounded-md border border-slate-800/70 bg-slate-900/60 px-3 py-1 text-xs text-slate-300 hover:text-slate-100"
            onClick={() => {
              const nextTitle = window.prompt("Rename thread", thread.title);
              if (nextTitle && nextTitle.trim()) {
                onRenameThread(nextTitle.trim());
              }
            }}
          >
            Rename
          </button>
          <button
            className="rounded-md border border-slate-800/70 bg-slate-900/60 px-3 py-1 text-xs text-rose-300 hover:text-rose-200"
            onClick={onClearThread}
          >
            Clear
          </button>
        </div>
      </div>
      <div className="flex flex-col gap-4">
        {thread.messages.length === 0 ? (
          <div className="rounded-2xl border border-slate-800/60 bg-slate-900/50 px-4 py-6 text-sm text-slate-400">
            Start by typing a prompt below. This thread will show the full conversation once you send a message.
          </div>
        ) : (
          thread.messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}
        {isTyping && (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="flex h-6 items-center gap-1 rounded-full bg-slate-900/60 px-3">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.2s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.1s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
            </span>
            Generating response…
          </div>
        )}
      </div>
    </div>
  );
}
