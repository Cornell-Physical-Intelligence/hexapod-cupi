"""Explicit SDF query hypotheses and proposed diagnostic bounds; no adoption."""
import itertools
import numpy as np

CELL_M = 0.00014988087787060067
BOUNDS = {'distance_rms_m': 2*CELL_M, 'distance_max_m': 4*CELL_M,
          'gradient_rms': .2, 'scale_slope_min': .85, 'scale_slope_max': 1.15,
          'hypothesis_score_margin': 2.0, 'repeat_max_abs': 0.0,'gradient_fd_max_abs':.25}


def finite(x, shape=None):
    x=np.asarray(x,dtype=np.float64)
    if (shape is not None and x.shape!=shape) or not np.isfinite(x).all():raise ValueError('Invalid native query tensor')
    return x


def hypotheses(raw, distance, gradient):
    raw=finite(raw);distance=finite(distance);gradient=finite(gradient)
    if raw.shape!=(len(distance),4) or gradient.shape!=(len(distance),3):raise ValueError('Wrong query shape')
    scores=[]
    for channel in range(4):
        for order in itertools.permutations([i for i in range(4)if i!=channel]):
            for sign in (1,-1):
                for scale in (1.,.001,1000.):
                    dr=float(np.sqrt(np.mean((sign*scale*raw[:,channel]-distance)**2)))
                    gr=float(np.sqrt(np.mean((sign*raw[:,order]-gradient)**2)))
                    scores.append({'distance_channel':channel,'gradient_channels':list(order),'sign':sign,'distance_scale':scale,
                                   'distance_rms_m':dr,'gradient_rms':gr,'score':dr/BOUNDS['distance_rms_m']+gr/BOUNDS['gradient_rms']})
    return sorted(scores,key=lambda r:r['score'])


def score(raw, repeated, fixture, world_hypothesis, link_hypothesis):
    n=len(fixture['points']);raw=finite(raw,(6,n,4));repeated=finite(repeated,raw.shape)
    expected=finite(fixture['distance_m']);grad=finite(fixture['gradient'])
    anchors=np.asarray(fixture['semantic_anchor_indices'],dtype=int)
    singular=np.linalg.svd(grad[anchors],compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>1e-12 else None
    if condition is None or condition>10 or not (np.any(expected[anchors]<-3*CELL_M) and np.any(expected[anchors]>3*CELL_M)):
        raise ValueError('Semantic anchors cannot identify normals and both signs')
    ranked=hypotheses(raw[:,anchors].reshape(-1,4),np.tile(expected[anchors],6),np.tile(grad[anchors],(6,1)))
    declared=next(r for r in ranked if r['distance_channel']==0 and r['gradient_channels']==[1,2,3] and r['sign']==1 and r['distance_scale']==1)
    competitors=[r for r in ranked if r is not declared]
    repeat_error=float(np.max(np.abs(raw-repeated)))
    sign_mask=np.abs(expected)>3*CELL_M
    sign_errors=int(np.sum(np.sign(raw[:,sign_mask,0])!=np.sign(expected[sign_mask])[None,:]))
    regions={}
    for label in sorted(set(fixture['regions'])):
        mask=np.asarray(fixture['regions'])==label;error=raw[:,mask,0]-expected[mask]
        regions[label]={'points_per_shape':int(mask.sum()),'distance_abs_p95_m':float(np.quantile(np.abs(error),.95)),
                        'distance_abs_max_m':float(np.abs(error).max()),'distance_rms_m':float(np.sqrt(np.mean(error**2)))}
    slopes=[];intercepts=[]
    for row in raw:
        fit=np.polyfit(expected[anchors],row[anchors,0],1)
        slopes.append(float(fit[0]));intercepts.append(float(fit[1]))
    # This compares the same inputs under a world-coordinate hypothesis. Far points
    # may exceed the native field's support; report that limitation, not a fake proof.
    world=finite(world_hypothesis,(6,len(anchors),4))
    link=finite(link_hypothesis,(6,len(anchors),4))
    local_rms=float(np.sqrt(np.mean((raw[:,anchors,0]-expected[anchors])**2)))
    frames={}
    expected_anchor=np.tile(np.c_[expected[anchors],grad[anchors]][None],(6,1,1))
    for name,alternative in [('link',link),('world',world)]:
        dr=float(np.sqrt(np.mean((raw[:,anchors,0]-alternative[:,:,0])**2)))
        gr=float(np.sqrt(np.mean((raw[:,anchors,1:]-alternative[:,:,1:])**2)))
        separation=float(np.sqrt(np.mean((expected_anchor[:,:,0]-alternative[:,:,0])**2)))/BOUNDS['distance_rms_m']
        separation+=float(np.sqrt(np.mean((expected_anchor[:,:,1:]-alternative[:,:,1:])**2)))/BOUNDS['gradient_rms']
        frames[name]={'distance_rms_m':dr,'gradient_rms':gr,'score':dr/BOUNDS['distance_rms_m']+gr/BOUNDS['gradient_rms'],
                      'source_prediction_separation_score':separation,'identifiable_against_shape_frame':separation>=BOUNDS['hypothesis_score_margin']}
    stencil=[]
    for item in fixture['derivative_stencils']:
        i=item['center'];pairs=item['pairs'];h=item['h_m']
        derivative=np.stack([(raw[:,pair[0],0]-raw[:,pair[1],0])/(2*h)for pair in pairs],axis=-1)
        stencil.append({'center':i,'distance_derivative':derivative.tolist(),
                        'raw_reported_gradient':raw[:,i,1:].tolist(),
                        'max_abs_discrepancy':float(np.max(np.abs(derivative-raw[:,i,1:])))})
    checks={'declared_layout_best':ranked[0] is declared,
            'declared_distance_rms':declared['distance_rms_m']<=BOUNDS['distance_rms_m'],
            'declared_gradient_rms':declared['gradient_rms']<=BOUNDS['gradient_rms'],
            'hypothesis_separated':competitors[0]['score']-declared['score']>=BOUNDS['hypothesis_score_margin'],
            'metric_scale_anchors':all(BOUNDS['scale_slope_min']<=s<=BOUNDS['scale_slope_max']for s in slopes),
            'anchor_sign':bool(np.all(np.sign(raw[:,anchors,0])==np.sign(expected[anchors])[None,:])),
            'shape_vs_link_frame_separated':frames['link']['identifiable_against_shape_frame'] and frames['link']['score']-declared['score']>=BOUNDS['hypothesis_score_margin'],
            'gradient_distance_fd':all(r['max_abs_discrepancy']<=BOUNDS['gradient_fd_max_abs']for r in stencil),
            'repeat_identical':repeat_error<=BOUNDS['repeat_max_abs']}
    return {'schema':'canonical_sdf_distance_diagnostic_v1','bounds':BOUNDS,'bounds_scope':'Proposed query diagnostic, not inherited hardware/physics gates',
            'semantics_checks':checks,'declared_semantics_supported':all(checks.values()),
            'failure_interpretation':'A mismatch can be ABI, frame, scale or cooked geometry. Alternative scores are diagnostic; never auto-adopt a permutation.',
            'declared':declared,'best_alternatives':competitors[:4],'sign_errors_beyond_3_nominal_cells':sign_errors,
            'regions':regions,'distance_scale_slopes':slopes,'distance_scale_intercepts_m':intercepts,'repeat_max_abs':repeat_error,'derivative_stencils':stencil,
            'anchor_normal_singular_values':singular.tolist(),'anchor_normal_condition':condition,
            'geometry_accuracy_within_proposed_bounds':sign_errors==0 and all(r['distance_abs_max_m']<=BOUNDS['distance_max_m'] for r in regions.values()),
            'frame_comparison':{'shape_local_input_oracle_rms_m':local_rms,'alternatives':frames,
               'scope':'Same inputs under shape, link and world interpretations at six current poses. Each alternative rotates its expected gradient too. Link-frame identifiability is required. World points may be outside the field; that comparison is diagnostic, not unrestricted frame covariance.'},
            'shape_to_shape_max_distance_spread_m':float(np.ptp(raw[:,:,0],axis=0).max()),
            'shape_to_shape_max_gradient_component_spread':float(np.ptp(raw[:,:,1:],axis=0).max()),
            'standing_admitted':False,'contact_admitted':False,'training_allowed':False}
