import re
from itertools import combinations
from collections import defaultdict

# store our data
data = []

with open(r"folder\CollaborativeAI\social_choice\00073-00000002.cat", "r") as f:
    for line in f:
        if ":" in line and not line.startswith('#'): # if we find a line with a colon it is a line with voting data

            # split into count and the actual ranking
            count, ranking_section = line.split(":")
            count = int(count.strip())

            # delimiter for groups 
            groups = re.findall(r'\{([^}]*)\}', ranking_section)

            ranking = []

            for group in groups:
                nums = []
                if group:
                    nums = [int(x) for x in group.split(",")]
                ranking.append(nums)
            
            data.append((count, ranking))

class VotingSystems:


    def plurality(data):

        # we only take into account the candidates that are ranked in the no.1 spot

        # map candidates to their count
        first_places = {}
        # iterate through all votes
        for count, ranking in data:

            if ranking:
                # take the 1st place ranking
                first = ranking[0]

                # if its non empty
                if first:
                    for f in first:
                        first_places[f] = count + first_places.get(f, 0)

        first_places = sorted(first_places.items(), key=lambda item: item[1], reverse=True)
        
        # in case we haev ties
        winners = [first_places[0][0]]
        i = 1
        while i < len(first_places) and first_places[i][1] == first_places[i - 1][1]:
            winners.append(first_places[i][0])
            i += 1

        print(first_places)
        print(f"The winning candidates for pluratity voting are {winners}")

    def anti_plurality(data):

        # we only take into account last places
        last_places = {}

        # iterate through
        for count, ranking in data:

            if ranking:
                
                # take last place
                last = ranking[-1]

                if last:
                    for l in last:
                        last_places[l] = count + last_places.get(l, 0)

                
        last_places = sorted(last_places.items(), key=lambda item: item[1])

        # in case we have ties
        winners = [last_places[0][0]]
        i = 1
        while i < len(last_places) and last_places[i][1] == last_places[i - 1][1]:
            winners.append(last_places[i][0])
            i += 1
        # print(last_places)
        print(f"The winning candidates for anti-plurality voting are {winners}")

    def borda(data):

        scores = {}

        # iterate through
        for count, ranking in data:

            for i, group in enumerate(ranking):

                for cand in group:

                    scores[cand] = ((count * (3 - i - 1))) + scores.get(cand, 0) # we know they can only be 1st, 2nd or 3rd

        # sort by highest score first
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # in case we haev ties
        winners = [scores[0][0]]
        i = 1
        while i < len(scores) and scores[i][1] == scores[i - 1][1]:
            winners.append(scores[i][0])
            i += 1
        # print(scores)
        print(f"The winning candidates for Borda voting are {winners}")

    def copeland(data):

        # make a set of candidates this is gonna be fixed as per the .cat file
        candidates = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        # initialise scores
        scores = {c : 0 for c in candidates}

        # go through all pairs

        for a, b in combinations(candidates, 2):

            a_wins = 0
            b_wins = 0

            ties = 0

            # go through all the votes
            for count, ranking in data:

                group_pos = {c : i for i, g in enumerate(ranking) for c in g}

                a_pos = group_pos.get(a, len(ranking))
                b_pos = group_pos.get(b, len(ranking))


                # if a is positioned higher than b 
                if a_pos < b_pos:
                    a_wins += count
                elif b_pos < a_pos:
                    b_wins += count
                else: # if there is a tie
                    ties += count 

            # update the scores
            # if a won more than b
            if a_wins > b_wins:
                scores[a] += 1
            elif b_wins > a_wins:
                scores[b] += 1
            else:
                scores[a] += 0.5
                scores[b] += 0.5

        max_score = max(scores.values())

        # get all the eligible winners
        winners = [c for c, s in scores.items() if s == max_score]
        print(f"The winning candidates for copeland voting are {winners}")
        

    def stv(data):

        # have a set of active candidates
        active = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        # create copy fo votes
        votes = list(data)

        while len(active) > 1:

            # count first preferences
            firsts = defaultdict(float)

            for count, ranking in votes:

                if ranking:

                    first_group = [c for c in ranking[0] if c in active]

                    if first_group:

                        weight = count

                        for c in first_group:
                            firsts[c] += weight
            
            # find candidate with lowest vote
            min_count = min(firsts[c] for c in active)
            elim = [c for c in active if firsts[c] == min_count]

            # incase of ties
            if len(elim) == len(active):
                break

            # remove candidates from rankings
            new_v = []
            for count, ranking in votes:
                new_ranking = []

                for group in ranking:
                    new_group = [c for c in group if c not in elim]

                    if new_group:
                        new_ranking.append(new_group)

                if new_ranking:
                    new_v.append((count, new_ranking))

            votes = new_v

            # update active candidates
            active -= set(elim)

        print(f"The winning candidates for Single Transferable Vote are {list(active)}")

        

 

                    







VotingSystems.plurality(data)   
VotingSystems.anti_plurality(data)
VotingSystems.borda(data)
VotingSystems.copeland(data)
VotingSystems.stv(data)

