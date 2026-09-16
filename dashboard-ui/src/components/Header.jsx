import React, { useState, useEffect } from 'react';
import { Volume2, VolumeX, Crosshair, Shield, PenTool } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';

export function Header({ stats, audioMuted, toggleAudio, isDrawing, setIsDrawing }) {
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      const d = new Date();
      setTimeStr(d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const pipe = stats?.pipeline || {};
  const zeroDceActive = pipe.zero_dce_active || false;

  return (
    <header className="flex h-16 items-center justify-between border bg-card px-6 shadow-sm mx-4 mt-4 rounded-2xl shrink-0">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-xl font-bold text-primary">
          <Crosshair className="h-6 w-6" />
          <span>INNOVISION</span>
        </div>
        <div className="ml-4 flex gap-3">
          <Badge variant="outline" className="text-muted-foreground">
            {timeStr}
          </Badge>
          <Badge variant="outline">
            FPS: {pipe.fps?.toFixed(1) || '0.0'}
          </Badge>
          <Badge variant="outline">
            TRACKS: {pipe.active_tracks || 0}
          </Badge>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <motion.div
          animate={zeroDceActive ? { opacity: [0.5, 1, 0.5] } : {}}
          transition={{ repeat: Infinity, duration: 2 }}
        >
          <Badge
            variant="outline"
            className={cn(
              "px-3 py-1",
              zeroDceActive
                ? "border-green-500 text-green-500 bg-green-500/10"
                : "border-primary text-primary bg-primary/10"
            )}
          >
            {zeroDceActive ? "ZERO-DCE: ACTIVE (BOOSTING)" : "ZERO-DCE: STANDBY"}
          </Badge>
        </motion.div>

        <Button
          onClick={() => setIsDrawing(!isDrawing)}
          variant={isDrawing ? "default" : "outline"}
          className={cn("transition-colors", isDrawing && "bg-primary text-primary-foreground hover:bg-primary/90")}
        >
          <PenTool className="mr-2 h-4 w-4" />
          {isDrawing ? "CANCEL DRAWING" : "DRAW FENCE"}
        </Button>

        <Button
          variant={audioMuted ? "destructive" : "outline"}
          size="sm"
          onClick={toggleAudio}
          className="w-32"
        >
          {audioMuted ? (
            <><VolumeX className="mr-2 h-4 w-4" /> MUTED</>
          ) : (
            <><Volume2 className="mr-2 h-4 w-4" /> AUDIO ON</>
          )}
        </Button>
      </div>
    </header>
  );
}
