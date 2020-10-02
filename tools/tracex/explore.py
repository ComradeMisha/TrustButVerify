"""TraceX entrypoint."""

import io
import sys

import flask
import numpy as np
import PIL

from alpacka import data as alpacka_data
from alpacka import tracing


app = flask.Flask(__name__, static_url_path='', static_folder='static')
trace = None
rendered_trajectory = None
entities = None


@app.route('/')
def index():
    return app.send_static_file('index.html')


def render_trajectory(trajectory, env_renderer):
    """Renders a trajectory into a collection of node entity dicts.

    The returned dicts are JSON-serializable.

    Entities are described in lazify_entity.

    Args:
        trajectory (tracing.Trajectory): Trajectory to render.
        env_renderer (envs.EnvRenderer): EnvRenderer to use.

    Returns:
        Pair (root, entities), where root is the root entity dict, and
        entities is a dict of node entities, indexed by id.
    """
    entities = {}

    def add_entity(entity):
        entity['id'] = len(entities)
        entities[entity['id']] = entity
        return entity

    states = [trajectory.init_state] + [
        transition.to_state for transition in trajectory.transitions
    ]
    state_passes = [
        transition.passes for transition in trajectory.transitions
    ] + [[]]

    def to_primitive(x):
        if isinstance(x, np.ndarray):
            if np.any(np.isnan(x)):
                return None
            else:
                return x.tolist()
        elif isinstance(x, np.number):
            return x.item()
        else:
            return x

    def render_state(state):
        return {
            'terminal': state.terminal,
            'state_info': alpacka_data.nested_map(
                to_primitive, state.state_info
            ),
        }

    def render_agent_info(agent_info):
        def render_entry(entry):
            if isinstance(entry, list):
                return {
                    env_renderer.render_action(action): value
                    for (action, value) in enumerate(entry)
                }

        return {
            key: render_entry(value) for (key, value) in agent_info.items()
        }

    def render_transition(transition):
        if transition is None:
            return {}
        else:
            return {
                'agent_info': render_agent_info(alpacka_data.nested_map(
                    to_primitive, transition.agent_info,
                )),
                'action': env_renderer.render_action(transition.action),
                'reward': to_primitive(transition.reward),
            }

    def render_pass(pass_, init):
        transition = pass_[0]
        data = {
            'type': 'model_init' if init else 'model',
            **render_state(transition.to_state),
            **render_transition(transition),
        }
        if len(pass_) > 1:
            data['children'] = [render_pass(pass_[1:], False)]
        return add_entity(data)

    def show_children_if_nonempty(children):
        if children:
            return {'children': children}
        else:
            return {}

    root = add_entity({
        'type': 'root',
        'children': [
            add_entity({
                'type': 'real',
                **show_children_if_nonempty([
                    render_pass(pass_, True) for pass_ in passes if pass_
                ]),
                **render_state(state),
                **render_transition(transition),
            })
            for (state, passes, transition) in zip(
                states, state_passes, [None] + trajectory.transitions
            )
        ]
    })
    return (root, entities)


def lazify_entity(entity, depth, lazy_keys):
    """Trims an entity to decrease its size.

    Entity is a JSON-serializable recursive data structure. This function trim
    the depth of this structure, leaving "stubs" that need to be requested via
    separate API calls. Trimmed entities get their "stub" key set to True.

    Args:
        entity: Entity to lazify.
        depth (int): Maximum depth.
        lazy_keys (list of str): Dict keys to recurse into. The others will
            be rewritten as-is.

    Returns:
        Lazified entity.
    """
    if isinstance(entity, dict):
        entity = {
            key: (
                value if key not in lazy_keys
                else lazify_entity(value, depth, lazy_keys)
            )
            for (key, value) in entity.items()
        }
        if depth == 0:
            entity['stub'] = True
        return entity
    elif isinstance(entity, (list, tuple)):
        if depth == 0:
            return type(entity)()
        else:
            return type(entity)(
                lazify_entity(x, depth - 1, lazy_keys) for x in entity
            )
    else:
        return entity


@app.route('/data')
def data():
    return flask.jsonify(
        lazify_entity(rendered_trajectory, depth=2, lazy_keys=('children',))
    )


@app.route('/entity/<int:entity_id>')
def entity(entity_id):
    return flask.jsonify(entities[entity_id])


@app.route('/render_state/<int:entity_id>')
def render_state(entity_id):
    rgb_array = trace.renderer.render_state(entities[entity_id]['state_info'])
    img = PIL.Image.fromarray(rgb_array)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    trace = tracing.load(sys.argv[1])
    (rendered_trajectory, entities) = render_trajectory(
        trace.trajectory, trace.renderer
    )
    app.run()
