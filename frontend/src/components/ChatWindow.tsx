import { Thread } from "@/data/threads";
import { MessageBubble } from "@/components/MessageBubble";

interface ChatWindowProps {
  thread: Thread;
}

export function ChatWindow({ thread }: ChatWindowProps) {
  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-6 py-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">{thread.title}</h1>
        <p className="text-xs text-slate-500">
          Demo thread showing model capability
        </p>
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
      </div>
    </div>
  );
}
