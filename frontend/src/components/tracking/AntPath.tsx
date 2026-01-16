import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import { antPath } from 'leaflet-ant-path';

interface AntPathProps {
  positions: [number, number][];
  options?: {
    color?: string;
    weight?: number;
    opacity?: number;
    paused?: boolean;
    reverse?: boolean;
    delay?: number;
    dashArray?: number[];
    pulseColor?: string;
  };
}

export default function AntPath({ positions, options }: AntPathProps) {
  const map = useMap();

  useEffect(() => {
    if (!positions || positions.length === 0) return;

    const path = antPath(positions, {
      "delay": 400,
      "dashArray": [10, 20],
      "weight": 5,
      "color": "#0000FF",
      "pulseColor": "#FFFFFF",
      "paused": false,
      "reverse": false,
      "hardwareAccelerated": true,
      ...options
    });

    path.addTo(map);

    return () => {
      map.removeLayer(path);
    };
  }, [map, positions, options]);

  return null;
}
