import { env, pipeline } from '@xenova/transformers';
import { useEffect, useState } from 'react';

// Skip local model checks since we are running in the browser
env.allowLocalModels = false;
env.useBrowserCache = true;

export function useRoofSegmenter() {
  const [model, setModel] = useState<any>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>(
    'loading'
  );

  useEffect(() => {
    async function loadAI() {
      try {
        // Load the "Segment Anything" Model (Quantized for Browser)
        // This downloads ~10-20MB the first time
        console.log('Loading AI Model...');
        const segmenter = await pipeline(
          'image-segmentation',
          'Xenova/slimsam-77-uniform',
          { device: 'webgpu' } // Tries GPU first, falls back to CPU
        );

        setModel(segmenter);
        setStatus('ready');
        console.log('AI Model Ready');
      } catch (e) {
        console.error('AI Load Failed:', e);
        setStatus('error');
      }
    }
    loadAI();
  }, []);

  const segmentClick = async (
    canvas: HTMLCanvasElement,
    clickX: number,
    clickY: number
  ) => {
    if (!model || status !== 'ready') return null;

    try {
      // 1. Snapshot the map
      const imageUrl = canvas.toDataURL('image/png');

      // 2. Run AI
      // We treat the click as a "Positive Point" prompt
      const output = await model(imageUrl, {
        points: [clickX, clickY],
        labels: [1], // 1 = "Include this area"
      });

      // Output is a binary mask (0s and 1s).
      // In a full app, you'd use marching squares to vectorize this.
      // For MVP, we just return the raw mask data.
      return output;
    } catch (e) {
      console.error('Segmentation failed:', e);
      return null;
    }
  };

  return { status, segmentClick };
}
