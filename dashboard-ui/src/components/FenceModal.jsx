import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';

export function FenceModal({ isOpen, onClose, onSave }) {
  const [name, setName] = useState('');
  const [zoneType, setZoneType] = useState('RESTRICTED_PERIMETER');
  const [loiter, setLoiter] = useState('5.0');

  const handleSave = () => {
    onSave({
      name: name.trim() || `Tactical Zone ${Date.now() % 1000}`,
      zoneType,
      loiter: parseFloat(loiter) || 5.0
    });
    setName('');
    setZoneType('RESTRICTED_PERIMETER');
    setLoiter('5.0');
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Configure Tactical Zone</DialogTitle>
          <DialogDescription>
            Set the parameters for the new virtual perimeter.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="text-right">
              Name
            </Label>
            <Input 
              id="name" 
              value={name} 
              onChange={(e) => setName(e.target.value)} 
              placeholder="e.g. Sector 7G" 
              className="col-span-3" 
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="zoneType" className="text-right">
              Type
            </Label>
            <select 
              id="zoneType" 
              value={zoneType} 
              onChange={(e) => setZoneType(e.target.value)}
              className="col-span-3 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <option value="RESTRICTED_PERIMETER">Strict Perimeter (Breach Alert)</option>
              <option value="LOITERING_ZONE">Loitering Zone (Time-based Alert)</option>
            </select>
          </div>
          {zoneType === 'LOITERING_ZONE' && (
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="loiter" className="text-right">
                Loiter (s)
              </Label>
              <Input 
                id="loiter" 
                type="number" 
                step="0.5" 
                value={loiter} 
                onChange={(e) => setLoiter(e.target.value)} 
                className="col-span-3" 
              />
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave}>Save Zone</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
