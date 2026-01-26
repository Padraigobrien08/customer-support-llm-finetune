import { CheckCircle2, AlertCircle, Info } from "lucide-react";
import { cn } from "@/lib/utils";

export interface QualityMetrics {
  score: number;
  grade: "A" | "B" | "C" | "D" | "F";
  issues: string[];
  strengths: string[];
  details?: {
    coherence?: number;
    repetition?: number;
    spacing?: number;
    completeness?: number;
    relevance?: number;
    length?: number;
  };
}

interface QualityScoreProps {
  quality: QualityMetrics;
  showDetails?: boolean;
  compact?: boolean;
}

export function QualityScore({ quality, showDetails = false, compact = false }: QualityScoreProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-emerald-300 bg-emerald-500/20";
    if (score >= 60) return "text-yellow-300 bg-yellow-500/20";
    return "text-red-300 bg-red-500/20";
  };

  const getGradeColor = (grade: string) => {
    if (grade === "A" || grade === "B") return "text-emerald-300";
    if (grade === "C") return "text-yellow-300";
    return "text-red-300";
  };

  if (compact) {
    return (
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
          getScoreColor(quality.score)
        )}
        title={`Quality: ${quality.score}% (${quality.grade})`}
      >
        {quality.score >= 70 ? (
          <CheckCircle2 className="h-3 w-3" />
        ) : (
          <AlertCircle className="h-3 w-3" />
        )}
        <span>{quality.score}%</span>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-800/70 bg-slate-900/50 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {quality.score >= 70 ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          ) : (
            <AlertCircle className="h-4 w-4 text-yellow-400" />
          )}
          <span className="text-sm font-semibold text-slate-200">Quality Assessment</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn("text-lg font-bold", getScoreColor(quality.score))}>
            {quality.score}%
          </span>
          <span className={cn("text-sm font-semibold", getGradeColor(quality.grade))}>
            ({quality.grade})
          </span>
        </div>
      </div>

      {showDetails && quality.details && (
        <div className="mb-2 space-y-1 text-xs">
          <div className="flex justify-between text-slate-400">
            <span>Coherence:</span>
            <span className="text-slate-300">{quality.details.coherence ?? "N/A"}%</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Repetition:</span>
            <span className="text-slate-300">{quality.details.repetition ?? "N/A"}%</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Spacing:</span>
            <span className="text-slate-300">{quality.details.spacing ?? "N/A"}%</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Completeness:</span>
            <span className="text-slate-300">{quality.details.completeness ?? "N/A"}%</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Relevance:</span>
            <span className="text-slate-300">{quality.details.relevance ?? "N/A"}%</span>
          </div>
        </div>
      )}

      {quality.strengths.length > 0 && (
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium text-emerald-300">Strengths:</div>
          <ul className="space-y-0.5 text-xs text-slate-400">
            {quality.strengths.map((strength, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <CheckCircle2 className="mt-0.5 h-3 w-3 flex-shrink-0 text-emerald-400" />
                <span>{strength}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {quality.issues.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-medium text-yellow-300">Issues:</div>
          <ul className="space-y-0.5 text-xs text-slate-400">
            {quality.issues.map((issue, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <AlertCircle className="mt-0.5 h-3 w-3 flex-shrink-0 text-yellow-400" />
                <span>{issue}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
