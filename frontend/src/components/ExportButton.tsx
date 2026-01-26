import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Thread } from "@/data/threads";

interface ExportButtonProps {
  thread: Thread;
  className?: string;
}

export function ExportButton({ thread, className }: ExportButtonProps) {
  const handleExport = () => {
    const exportData = {
      thread: {
        id: thread.id,
        title: thread.title,
        exportedAt: new Date().toISOString(),
        messageCount: thread.messages.length
      },
      messages: thread.messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `conversation-${thread.id}-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportMarkdown = () => {
    const markdown = `# ${thread.title}\n\n` +
      `**Exported:** ${new Date().toLocaleString()}\n` +
      `**Messages:** ${thread.messages.length}\n\n` +
      "---\n\n" +
      thread.messages.map((msg) => {
        const role = msg.role === "user" ? "**User**" : "**Assistant**";
        return `${role} (${msg.timestamp || "Unknown time"})\n\n${msg.content}\n\n---\n`;
      }).join("\n");

    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `conversation-${thread.id}-${new Date().toISOString().split("T")[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative group">
      <Button
        variant="ghost"
        size="icon"
        onClick={handleExport}
        className={className}
        title="Export conversation"
      >
        <Download className="h-4 w-4" />
      </Button>
      <div className="absolute right-0 top-full mt-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
        <div className="rounded-lg border border-slate-800/70 bg-slate-950 shadow-xl p-1 min-w-[140px]">
          <button
            onClick={handleExport}
            className="w-full rounded-md px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-900/70"
          >
            Export as JSON
          </button>
          <button
            onClick={handleExportMarkdown}
            className="w-full rounded-md px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-900/70"
          >
            Export as Markdown
          </button>
        </div>
      </div>
    </div>
  );
}
