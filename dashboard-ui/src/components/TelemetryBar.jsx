import React from 'react';
import { Progress } from './ui/progress';
import { Cpu, HardDrive, MemoryStick } from 'lucide-react';

export function TelemetryBar({ stats }) {
  if (!stats) return null;

  const { gpu, system } = stats;

  return (
    <div className="flex h-16 items-center justify-between border-t bg-card px-8 shadow-sm text-sm text-muted-foreground">
      <div className="flex items-center gap-6 flex-1">
        
        {/* GPU Telemetry */}
        <div className="flex items-center gap-4 flex-1 max-w-sm">
          <HardDrive className="h-5 w-5 text-primary" />
          <div className="flex-1 space-y-1">
            <div className="flex justify-between font-geist">
              <span>GPU</span>
              <span>{gpu?.vram_used_mb} / {gpu?.vram_total_mb} MB ({gpu?.vram_percent}%)</span>
            </div>
            <Progress value={gpu?.vram_percent || 0} className="h-1.5" />
          </div>
        </div>

        {/* CPU Telemetry */}
        <div className="flex items-center gap-4 flex-1 max-w-sm">
          <Cpu className="h-5 w-5 text-primary" />
          <div className="flex-1 space-y-1">
            <div className="flex justify-between font-geist">
              <span>CPU</span>
              <span>{system?.cpu_percent || 0}%</span>
            </div>
            <Progress value={system?.cpu_percent || 0} className="h-1.5" />
          </div>
        </div>

        {/* RAM Telemetry */}
        <div className="flex items-center gap-4 flex-1 max-w-sm">
          <MemoryStick className="h-5 w-5 text-primary" />
          <div className="flex-1 space-y-1">
            <div className="flex justify-between font-geist">
              <span>RAM</span>
              <span>{system?.ram_used_gb} / {system?.ram_total_gb} GB ({system?.ram_percent}%)</span>
            </div>
            <Progress value={system?.ram_percent || 0} className="h-1.5" />
          </div>
        </div>

      </div>
    </div>
  );
}
