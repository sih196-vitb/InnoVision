import React from 'react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Crosshair, ShieldAlert, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export function AlertFeed({ alerts, onClear }) {
  return (
    <div className="flex h-full flex-col bg-card/50 border border-zinc-800 rounded-xl shadow-lg">
      <div className="flex items-center justify-between border-b p-4">
        <h2 className="text-lg font-bold text-primary flex items-center gap-2">
          <ShieldAlert className="h-5 w-5" />
          INCIDENT LOG
        </h2>
        <div className="flex items-center gap-2">
          <Badge variant="outline">{alerts.length} ALERTS</Badge>
          {alerts.length > 0 && (
            <Button variant="ghost" size="icon" onClick={onClear} title="Clear Alerts" className="h-6 w-6 text-muted-foreground hover:text-destructive">
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-hide p-4 space-y-4">
        <AnimatePresence>
          {alerts.length === 0 ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-40 text-muted-foreground"
            >
              <ShieldAlert className="h-10 w-10 mb-2 opacity-20" />
              <p>Awaiting tactical incident events...</p>
            </motion.div>
          ) : (
            alerts.map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: -20, height: 0 }}
                animate={{ opacity: 1, x: 0, height: 'auto' }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2 }}
              >
                <Card className={`overflow-hidden border-l-4 ${
                  alert.threat_level === 'CRITICAL' ? 'border-l-destructive' :
                  alert.threat_level === 'INFO' ? 'border-l-blue-500' :
                  'border-l-orange-500'
                }`}>
                  <CardContent className="p-0 flex items-stretch">
                    <div className="w-24 bg-muted/30 flex-shrink-0 flex items-center justify-center border-r">
                      {alert.crop_base64 ? (
                        <img 
                          src={`data:image/jpeg;base64,${alert.crop_base64}`} 
                          alt="Offender" 
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <Crosshair className="h-6 w-6 text-muted-foreground" />
                      )}
                    </div>
                    <div className="p-3 flex-1 flex flex-col justify-center">
                      <div className="flex justify-between items-start mb-1">
                        <Badge 
                          variant={alert.threat_level === 'CRITICAL' ? 'destructive' : 'secondary'}
                          className={alert.threat_level === 'HIGH' ? 'bg-orange-500/20 text-orange-500 hover:bg-orange-500/30' : ''}
                        >
                          {alert.threat_level}
                        </Badge>
                        <span className="text-xs text-muted-foreground font-mono">
                          {alert.datetime_str ? alert.datetime_str.split(' ')[1] : ''}
                        </span>
                      </div>
                      <h4 className="font-semibold text-sm">{alert.alert_type}</h4>
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                        {alert.message}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
