class OccupancySmoother:

    def __init__(self, confirmation_threshold=2):

        self.confirmation_threshold = (
            confirmation_threshold
        )

        self.confirmed_state = None

        self.candidate_counts = {}

    def smooth(self, occupancy_map):

        # first frame initializes state
        if self.confirmed_state is None:

            self.confirmed_state = occupancy_map.copy()

            return self.confirmed_state

        for square in occupancy_map:

            current_value = occupancy_map[square]

            confirmed_value = (
                self.confirmed_state[square]
            )

            # same as confirmed state
            if current_value == confirmed_value:

                self.candidate_counts[square] = 0

                continue

            # candidate change
            self.candidate_counts[square] = (
                self.candidate_counts.get(square, 0) + 1
            )

            # confirm change
            if (
                self.candidate_counts[square]
                >= self.confirmation_threshold
            ):

                self.confirmed_state[square] = (
                    current_value
                )

                self.candidate_counts[square] = 0

        return self.confirmed_state.copy()
