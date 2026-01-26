import { ChevronDown, Cpu } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ModelInfo {
  id: string;
  name: string;
  adapterPath: string;
  modelId: string;
  description?: string;
}

const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: "qwen",
    name: "Qwen2.5-7B-Instruct",
    adapterPath: "outputs/run_002_qwen7b",
    modelId: "Qwen/Qwen2.5-7B-Instruct",
    description: "Currently loaded model"
  },
  {
    id: "mistral",
    name: "Mistral-7B-Instruct",
    adapterPath: "outputs/run_004_mistral7b",
    modelId: "mistralai/Mistral-7B-Instruct-v0.2",
    description: "Training in progress"
  },
  {
    id: "llama3",
    name: "Llama-3-8B-Instruct",
    adapterPath: "outputs/run_003_llama3",
    modelId: "meta-llama/Meta-Llama-3-8B-Instruct",
    description: "Training in progress"
  }
];

interface ModelSelectorProps {
  currentModelId?: string;
  onModelChange?: (model: ModelInfo) => void;
}

export function ModelSelector({ currentModelId, onModelChange }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const currentModel = AVAILABLE_MODELS.find((m) => m.id === currentModelId) || AVAILABLE_MODELS[0];

  const handleSelect = (model: ModelInfo) => {
    onModelChange?.(model);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <Button
        variant="ghost"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-slate-300 hover:text-slate-100"
      >
        <Cpu className="h-4 w-4" />
        <span className="text-sm font-medium">{currentModel.name}</span>
        <ChevronDown className={cn("h-3 w-3 transition-transform", isOpen && "rotate-180")} />
      </Button>
      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute right-0 top-full z-50 mt-2 w-64 rounded-lg border border-slate-800/70 bg-slate-950 shadow-xl">
            <div className="p-2">
              <div className="mb-2 px-3 py-2 text-xs font-semibold text-slate-400">
                Available Models
              </div>
              {AVAILABLE_MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => handleSelect(model)}
                  className={cn(
                    "w-full rounded-md px-3 py-2 text-left text-sm transition-colors",
                    "hover:bg-slate-900/70",
                    currentModel.id === model.id
                      ? "bg-slate-900/50 text-slate-100"
                      : "text-slate-300"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{model.name}</div>
                      {model.description && (
                        <div className="text-xs text-slate-500">{model.description}</div>
                      )}
                    </div>
                    {currentModel.id === model.id && (
                      <div className="h-2 w-2 rounded-full bg-emerald-400" />
                    )}
                  </div>
                </button>
              ))}
            </div>
            <div className="border-t border-slate-800/70 p-2">
              <div className="rounded-md bg-slate-900/50 px-3 py-2 text-xs text-slate-400">
                <div className="font-medium text-slate-300 mb-1">Note:</div>
                <div>Model switching requires backend restart with new MODEL_ID and ADAPTER_DIR environment variables.</div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
