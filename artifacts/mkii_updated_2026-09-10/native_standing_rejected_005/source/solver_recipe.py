"""Controlled velocity-iteration diagnostic; no geometry/servo/gate changes."""
EXPECTED_PARENT=(32,1)
CANDIDATE=(32,4)

def read(stage,roots,schema):
    result={}
    for root in roots:
        prim=stage.GetPrimAtPath(root+'/body')
        api=schema.PhysxArticulationAPI(prim)
        attrs=[api.GetSolverPositionIterationCountAttr(),api.GetSolverVelocityIterationCountAttr()]
        values=[a.Get() for a in attrs]
        if any(type(v)is not int for v in values):raise ValueError('Missing/noninteger articulation solver readback')
        result[root]={'position_iterations':values[0],'velocity_iterations':values[1],
                      'authored':[bool(a.HasAuthoredValueOpinion())for a in attrs]}
    return result

def configure(stage,roots,schema):
    before=read(stage,roots,schema)
    # Validate the whole parent scene before any authored mutation.
    if any((v['position_iterations'],v['velocity_iterations'])!=EXPECTED_PARENT or any(v['authored'])for v in before.values()):
        raise ValueError('Installed parent solver defaults differ from reviewed32/1 hypothesis: '+repr(before))
    for root in roots:
        api=schema.PhysxArticulationAPI(stage.GetPrimAtPath(root+'/body'))
        api.CreateSolverPositionIterationCountAttr(CANDIDATE[0])
        api.CreateSolverVelocityIterationCountAttr(CANDIDATE[1])
    after=verify(stage,roots,schema)
    return {'before':before,'after_authoring':after,
            'scope':'USD articulation attribute readback; not independent backend solver introspection',
            'hypothesis':'Additional velocity iterations may improve contact/velocity convergence; no repair is assumed.'}

def verify(stage,roots,schema):
    actual=read(stage,roots,schema)
    if any((v['position_iterations'],v['velocity_iterations'])!=CANDIDATE or not all(v['authored'])for v in actual.values()):
        raise ValueError('Authored diagnostic solver recipe changed')
    return actual
