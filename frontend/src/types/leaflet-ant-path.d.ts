declare module 'leaflet-ant-path' {
    import { Polyline, PolylineOptions } from 'leaflet';

    interface AntPathOptions extends PolylineOptions {
        delay?: number;
        dashArray?: number[] | string;
        pulseColor?: string;
        paused?: boolean;
        reverse?: boolean;
        hardwareAccelerated?: boolean;
    }

    export function antPath(
        latlngs: any[],
        options?: AntPathOptions
    ): any; 
}
