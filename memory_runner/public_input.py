class ReplayInput:
    def __init__(self, agent, specs):
        self.agent = agent
        self.specs = specs
        self.events = []
        self.seen = []
        self.targets = {}
        self.enemy_elixir = None
        self.last_tick = None

    def card(self, card_id):
        from rl.banana_policy import _native_card
        spec = self.specs.get(card_id)
        if spec is None:
            return f'unknown-{card_id}'
        compact = ''.join(c.lower() for c in spec.name if c.isalnum())
        return self.agent.native_name_to_policy.get(compact, _native_card(spec.name, self.agent.vocab.name_to_id))

    def update(self, raw, observation):
        tick = observation.tick
        if self.enemy_elixir is None:
            self.enemy_elixir = observation.players[0].elixir_exact
            self.last_tick = tick
        self.enemy_elixir = min(10., self.enemy_elixir + (tick-self.last_tick)*.05*observation.time.elixir_multiplier/2.8)
        self.last_tick = tick
        for event in raw['events']:
            card_id, owner = int(event['cardId']), int(event['owner'])
            card = self.card(card_id)
            if owner == 1:
                if card not in self.seen:
                    self.seen.append(card)
                self.enemy_elixir = max(0., self.enemy_elixir-int(event['cost']))
            if owner == 0 and card_id in self.targets:
                x, y = self.targets.pop(card_id)
            else:
                public_object = next((o for o in raw['objects'] if o['owner']==owner and o['cardId']==card_id), None)
                if public_object is None:
                    continue
                x, y = public_object['x'], public_object['y']
            self.events.append(dict(seconds=int(event['tick'])/20, side='team' if owner==0 else 'opponent',
                                    event_type='card_play', card=card, x=x, y=y))
