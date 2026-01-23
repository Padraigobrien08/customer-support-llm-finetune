import { Settings, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export interface GenerationSettings {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
}

interface SettingsPanelProps {
  settings: GenerationSettings;
  onSettingsChange: (settings: GenerationSettings) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  isOpen,
  onToggle
}: SettingsPanelProps) {
  const updateSetting = <K extends keyof GenerationSettings>(
    key: K,
    value: GenerationSettings[K]
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        onClick={onToggle}
        className="text-slate-400 hover:text-slate-200"
        title="Settings"
      >
        <Settings className="h-4 w-4" />
      </Button>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="relative w-full max-w-md rounded-lg border border-slate-800/70 bg-slate-950 p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-100">Generation Settings</h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggle}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-slate-300">
                  Max New Tokens
                </label>
                <Input
                  type="number"
                  min="50"
                  max="1000"
                  step="50"
                  value={settings.maxNewTokens}
                  onChange={(e) =>
                    updateSetting("maxNewTokens", parseInt(e.target.value) || 250)
                  }
                  className="bg-slate-900/50 text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500">
                  Maximum number of tokens to generate (50-1000)
                </p>
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-slate-300">
                  Temperature
                </label>
                <Input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) =>
                    updateSetting("temperature", parseFloat(e.target.value) || 0.6)
                  }
                  className="bg-slate-900/50 text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500">
                  Controls randomness (0 = deterministic, 2 = very creative)
                </p>
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-slate-300">
                  Top P
                </label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={settings.topP}
                  onChange={(e) =>
                    updateSetting("topP", parseFloat(e.target.value) || 0.85)
                  }
                  className="bg-slate-900/50 text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500">
                  Nucleus sampling threshold (0-1)
                </p>
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-slate-300">
                  Repetition Penalty
                </label>
                <Input
                  type="number"
                  min="1"
                  max="2"
                  step="0.1"
                  value={settings.repetitionPenalty}
                  onChange={(e) =>
                    updateSetting("repetitionPenalty", parseFloat(e.target.value) || 1.5)
                  }
                  className="bg-slate-900/50 text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500">
                  Penalty for repeating tokens (1 = no penalty, 2 = strong penalty)
                </p>
              </div>
            </div>
            <div className="mt-6 flex justify-end gap-2">
              <Button
                variant="secondary"
                onClick={() => {
                  onSettingsChange({
                    maxNewTokens: 250,
                    temperature: 0.6,
                    topP: 0.85,
                    repetitionPenalty: 1.5
                  });
                }}
              >
                Reset to Defaults
              </Button>
              <Button onClick={onToggle}>Done</Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
