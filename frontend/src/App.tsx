import { useMemo, useState } from "react";
import { ThreadList } from "@/components/ThreadList";
import { ChatWindow } from "@/components/ChatWindow";
import { PromptComposer } from "@/components/PromptComposer";
import { examplePrompts, initialThreads, Thread } from "@/data/threads";

const randomPrompt = () => {
  return examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
};

export default function App() {
  const [threads, setThreads] = useState<Thread[]>(initialThreads);
  const [activeId, setActiveId] = useState(initialThreads[0]?.id ?? "");
  const [input, setInput] = useState("");
  const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeId) || threads[0],
    [threads, activeId]
  );

  const handleSend = () => {
    if (!input.trim() || !activeThread) return;

    const newMessage = {
      id: `m-${Date.now()}`,
      role: "user" as const,
      content: input.trim(),
      timestamp: "Just now"
    };

    const placeholderResponse = {
      id: `m-${Date.now() + 1}`,
      role: "assistant" as const,
      content:
        "Thanks for the prompt. The UI is ready, but the model response pipeline isn't connected yet.",
      timestamp: "Just now"
    };

    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThread.id
          ? {
              ...thread,
              messages: [...thread.messages, newMessage, placeholderResponse]
            }
          : thread
      )
    );

    setInput("");
  };

  const handleGeneratePrompt = () => {
    setIsGeneratingPrompt(true);
    setTimeout(() => {
      setInput(randomPrompt());
      setIsGeneratingPrompt(false);
    }, 300);
  };

  const handleNewThread = () => {
    const nextIndex = threads.length + 1;
    const newThread: Thread = {
      id: `thread-${nextIndex}`,
      title: `New Thread ${nextIndex}`,
      messages: [
        {
          id: `m-${Date.now()}`,
          role: "assistant",
          content:
            "Start a new conversation to showcase a capability or test a prompt.",
          timestamp: "Just now"
        }
      ]
    };

    setThreads((prev) => [newThread, ...prev]);
    setActiveId(newThread.id);
  };

  if (!activeThread) {
    return null;
  }

  return (
    <div className="flex h-screen w-full bg-background text-foreground">
      <ThreadList
        threads={threads}
        activeId={activeId}
        onSelect={setActiveId}
        onNewThread={handleNewThread}
      />
      <div className="flex flex-1 flex-col">
        <ChatWindow thread={activeThread} />
        <PromptComposer
          value={input}
          onChange={setInput}
          onSend={handleSend}
          onGeneratePrompt={handleGeneratePrompt}
          isGenerating={isGeneratingPrompt}
        />
      </div>
    </div>
  );
}
