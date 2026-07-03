import type { Quality } from './types';

export interface ModelSelector {
  filterType: string; // 'rt'
  quality: Quality;
  hdr: boolean;
  useColor: boolean;
  useAlbedo: boolean;
  useNormal: boolean;
  cleanAux: boolean;
  dirtyAux: boolean;
}

// Resolve the OIDN model name + input channel count from the selector.
// Port of the old denoiserUtils.determineTensorMap — the resulting name maps
// directly to a converted ONNX file (e.g. rt_ldr_alb_nrm_small.onnx).
export function determineModel(props: ModelSelector): { name: string; channels: number } {
  let name = props.filterType; // 'rt'
  name += props.hdr ? '_hdr' : '_ldr';

  // cleanAux requires BOTH albedo and normal
  if (props.useAlbedo && props.useNormal && props.cleanAux) name += '_calb_cnrm';
  else {
    name += props.useAlbedo ? '_alb' : '';
    name += props.useNormal ? '_nrm' : '';
  }

  // quality -> size suffix (only available for some variants)
  const hasSmall = ['rt_hdr', 'rt_ldr', 'rt_hdr_alb', 'rt_ldr_alb', 'rt_hdr_alb_nrm',
    'rt_ldr_alb_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];
  const hasLarge = ['rt_alb', 'rt_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];
  let size: 'small' | 'large' | 'default' = 'default';
  if (props.quality === 'fast' && hasSmall.includes(name)) size = 'small';
  else if (props.quality === 'high' && hasLarge.includes(name)) size = 'large';
  if (size !== 'default') name += `_${size}`;

  let channels = 0;
  if (props.useColor) channels += 3;
  if (props.useAlbedo) channels += 3;
  if (props.useNormal) channels += 3;

  return { name, channels };
}
