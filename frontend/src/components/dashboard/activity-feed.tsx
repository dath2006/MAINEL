'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';

interface Event {
  type: string;
  data: unknown;
  timestamp: Date;
}

interface ActivityFeedProps {
  events: Event[];
}

function getEventIcon(type: string): string {
  switch (type) {
    case 'detection':
      return '🎯';
    case 'reid_match':
      return '🔗';
    case 'track_start':
      return '▶️';
    case 'track_end':
      return '⏹️';
    case 'transit':
      return '🚶';
    default:
      return '📍';
  }
}

function getEventBadgeVariant(type: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (type) {
    case 'reid_match':
      return 'default';
    case 'detection':
      return 'secondary';
    case 'track_end':
      return 'outline';
    default:
      return 'secondary';
  }
}

export function ActivityFeed({ events = [] }: ActivityFeedProps) {
  return (
    <Card className="h-[400px]">
      <CardHeader>
        <CardTitle className="text-sm font-medium">Activity Feed</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[320px] px-4">
          {events.length === 0 ? (
            <p className="py-4 text-center text-sm text-muted-foreground">
              No recent activity
            </p>
          ) : (
            <div className="space-y-3">
              {events.map((event, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 rounded-lg border p-3"
                >
                  <span className="text-lg">{getEventIcon(event.type)}</span>
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <Badge variant={getEventBadgeVariant(event.type)}>
                        {event.type}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {event.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {JSON.stringify(event.data).slice(0, 100)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
