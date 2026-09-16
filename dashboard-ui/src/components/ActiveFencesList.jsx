import React from 'react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Layers, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export function ActiveFencesList({ fences, onDeleteFence }) {
  if (!fences || fences.length === 0) return null;

  return (
    <div className="flex flex-col bg-card/50 border border-zinc-800 rounded-xl shadow-lg shrink-0 max-h-64">
      <div className="flex items-center justify-between border-b p-3">
        <h2 className="text-sm font-bold text-primary flex items-center gap-2">
          <Layers className="h-4 w-4" />
          ACTIVE ZONES
        </h2>
        <Badge variant="outline" className="text-xs">{fences.length}</Badge>
      </div>

      <div className="overflow-y-auto scrollbar-hide p-3 space-y-2">
        <AnimatePresence>
          {fences.map((fence) => (
            <motion.div
              key={fence.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <Card className="overflow-hidden border border-zinc-800 bg-black/40">
                <CardContent className="p-2 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: fence.color }} />
                    <div className="flex flex-col">
                      <span className="font-mono text-xs font-bold">{fence.name}</span>
                      <span className="text-[10px] text-muted-foreground">{fence.zone_type}</span>
                    </div>
                  </div>
                  <button 
                    onClick={() => onDeleteFence(fence.id)} 
                    className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-md transition-colors"
                    title="Delete Fence"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
