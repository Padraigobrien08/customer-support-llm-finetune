import { useEffect, useMemo, useState } from "react";
import { ThreadList } from "@/components/ThreadList";
import { ChatWindow } from "@/components/ChatWindow";
import { PromptComposer } from "@/components/PromptComposer";
import { HeaderBar } from "@/components/HeaderBar";
import { SettingsPanel, type GenerationSettings } from "@/components/SettingsPanel";
import { examplePrompts, initialThreads, Thread } from "@/data/threads";
import { generateMockReply } from "@/lib/mockModel";
import { checkModelHealth, generateModelReply } from "@/lib/modelApi";

const randomPrompt = () => {
  return examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
};

const generateThreadTitle = (firstMessage: string): string => {
  // Clean up the message and truncate to a reasonable length
  const cleaned = firstMessage.trim();
  const maxLength = 60;
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  // Truncate at the last space before maxLength to avoid cutting words
  const truncated = cleaned.substring(0, maxLength);
  const lastSpace = truncated.lastIndexOf(" ");
  return lastSpace > 0 ? truncated.substring(0, lastSpace) + "..." : truncated + "...";
};

export default function App() {
  const [threads, setThreads] = useState<Thread[]>(initialThreads);
  const [activeId, setActiveId] = useState(initialThreads[0]?.id ?? "");
  const [input, setInput] = useState("");
  const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [threadSearch, setThreadSearch] = useState("");
  const [modelStatus, setModelStatus] = useState<"connected" | "disconnected">(
    "disconnected"
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [generationSettings, setGenerationSettings] = useState<GenerationSettings>({
    maxNewTokens: 250,
    temperature: 0.6,
    topP: 0.85,
    repetitionPenalty: 1.5
  });

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
  useEffect(() => {
    let isMounted = true;
    checkModelHealth()
      .then((isHealthy) => {
        if (isMounted) {
          setModelStatus(isHealthy ? "connected" : "disconnected");
        }
      })
      .catch(() => {
        if (isMounted) {
          setModelStatus("disconnected");
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeId) || threads[0],
    [threads, activeId]
  );

  const handleSend = async () => {
    if (!input.trim() || !activeThread) return;

    const newMessage = {
      id: `m-${Date.now()}`,
      role: "user" as const,
      content: input.trim(),
      timestamp: "Just now"
    };

    // Auto-rename thread if this is the first message
    const isFirstMessage = activeThread.messages.length === 0;
    const shouldAutoRename = isFirstMessage && activeThread.title.startsWith("New Thread");

    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThread.id
          ? {
              ...thread,
              messages: [...thread.messages, newMessage],
              ...(shouldAutoRename && { title: generateThreadTitle(input.trim()) })
            }
          : thread
      )
    );

    setIsTyping(true);
    const messagesForModel = [...activeThread.messages, newMessage];
    let replyContent = "";

    try {
      replyContent = await generateModelReply(messagesForModel, generationSettings);
      setModelStatus("connected");
    } catch {
      replyContent = generateMockReply(input.trim(), activeThread);
      setModelStatus("disconnected");
    }

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

  const handleClearThread = () => {
    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThread.id ? { ...thread, messages: [] } : thread
      )
    );
  };

  const handleRenameThread = (title: string) => {
    setThreads((prev) =>
      prev.map((thread) =>
        thread.id === activeThread.id ? { ...thread, title } : thread
      )
    );
  };

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
          settingsButton={
            <SettingsPanel
              settings={generationSettings}
              onSettingsChange={setGenerationSettings}
              isOpen={settingsOpen}
              onToggle={() => setSettingsOpen((prev) => !prev)}
            />
          }
        />
        <ChatWindow
          thread={activeThread}
          isTyping={isTyping}
          onClearThread={handleClearThread}
          onRenameThread={handleRenameThread}
          examplePrompts={examplePrompts}
          onSelectPrompt={(prompt) => setInput(prompt)}
        />
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
