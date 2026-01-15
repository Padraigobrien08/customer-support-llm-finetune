import { useMemo, useState } from "react";
import { ThreadList } from "@/components/ThreadList";
import { ChatWindow } from "@/components/ChatWindow";
import { PromptComposer } from "@/components/PromptComposer";
import { HeaderBar } from "@/components/HeaderBar";
import { examplePrompts, initialThreads, Thread } from "@/data/threads";
import { generateMockReply } from "@/lib/mockModel";

const randomPrompt = () => {
  return examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
};

export default function App() {
  const [threads, setThreads] = useState<Thread[]>(initialThreads);
  const [activeId, setActiveId] = useState(initialThreads[0]?.id ?? "");
  const [input, setInput] = useState("");
  const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [threadSearch, setThreadSearch] = useState("");

  const filteredThreads = useMemo(() => {
    if (!threadSearch.trim()) return threads;
    const query = threadSearch.toLowerCase();
    return threads.filter((thread) => {
      const lastMessage = thread.messages[thread.messages.length - 1]?.content || "";
      return (
        thread.title.toLowerCase().includes(query) ||
        lastMessage.toLowerCase().includes(query)
      );
    });
  }, [threads, threadSearch]);
  const modelStatus: "connected" | "disconnected" = "disconnected";

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

    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThread.id
          ? {
              ...thread,
              messages: [...thread.messages, newMessage]
            }
          : thread
      )
    );

    setIsTyping(true);
    const replyContent = generateMockReply(input.trim(), activeThread);
    setTimeout(() => {
      const placeholderResponse = {
        id: `m-${Date.now() + 1}`,
        role: "assistant" as const,
        content: replyContent,
        timestamp: "Just now"
      };

      setThreads((prev) =>
        prev.map((thread) =>
          thread.id === activeThread.id
            ? {
                ...thread,
                messages: [...thread.messages, placeholderResponse],
                scriptedReplies: thread.scriptedReplies?.slice(1)
              }
            : thread
        )
      );
      setIsTyping(false);
    }, 600);

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
      messages: []
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
        threads={filteredThreads}
        activeId={activeId}
        onSelect={setActiveId}
        onNewThread={handleNewThread}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        searchValue={threadSearch}
        onSearchChange={setThreadSearch}
      />
      <div className="flex flex-1 flex-col">
        <HeaderBar
          modelName="TinyLlama + LoRA Adapter"
          status={modelStatus}
          environment="Local Demo"
        />
        <ChatWindow thread={activeThread} isTyping={isTyping} />
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
