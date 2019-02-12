import math

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,
                      "Norah Jones": 4.5, "Phoenix": 5.0,
                      "Slightly Stoopid": 1.5,
                      "The Strokes": 2.5, "Vampire Weekend": 2.0},

         "Bill": {"Blues Traveler": 2.0, "Broken Bells": 3.5,
                  "Deadmau5": 4.0, "Phoenix": 2.0,
                  "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},

         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,
                  "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5,
                  "Slightly Stoopid": 1.0},

         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,
                 "Deadmau5": 4.5, "Phoenix": 3.0,
                 "Slightly Stoopid": 4.5, "The Strokes": 4.0,
                 "Vampire Weekend": 2.0},

         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,
                    "Norah Jones": 4.0, "The Strokes": 4.0,
                    "Vampire Weekend": 1.0},

         "Jordyn": {"Broken Bells": 4.5, "Deadmau5": 4.0,
                    "Norah Jones": 5.0, "Phoenix": 5.0,
                    "Slightly Stoopid": 4.5, "The Strokes": 4.0,
                    "Vampire Weekend": 4.0},

         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,
                 "Norah Jones": 3.0, "Phoenix": 5.0,
                 "Slightly Stoopid": 4.0, "The Strokes": 5.0},

         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,
                      "Phoenix": 4.0, "Slightly Stoopid": 2.5,
                      "The Strokes": 3.0}
         }


def manhattan(rat1, rat2):
    distance = 0
    for key in rat1:
        if key in rat2:
            distance += abs(rat1[key] - rat2[key])
    return distance


def nearestNeighbor(username, users):
    distances = []
    for user in users:
        if user != username:
            distance = manhattan(users[user], users[username])
            distances.append((distance, user))
    distances.sort()
    return distances


def recommned(username, users):
    nearest = nearestNeighbor(username, users)[0][1]  # first find the nearest neighbor
    recommendation = []
    neighborRatings = users[nearest]
    userRatings = users[username]  # reference
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendation.append((artist,neighborRatings[artist]))
    # sort is more efficient
    return sorted(recommendation, key=lambda artistTuple: artistTuple[1], reverse=True)


def minkowski(rat1, rat2, r):
    distance = 0
    commonRatings = False
    for key in rat1:
        if key in rat2:
            distance += pow(abs(rat1[key] - rat2[key]), r)
    if commonRatings:
        return pow(distance, 1/r)
    else:
        return 0


def pearson(rat1, rat2):
    sum_xy, sum_x, sum_y, sum_x2, sum_y2 = 0
    n = 0
    for key in rat1:
        if key in rat2:
            n += 1
            x = rat1[key]
            y = rat2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += x**2
            sum_y2 += y**2

    if n == 0:
        return 0

    denominator = math.sqrt(sum_x2 - (sum_x**2) / n) * math.sqrt(sum_y2 - (sum_y**2) / n)
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) / n) / denominator










if __name__ == '__main__':
    # print(manhattan(users["Sam"], users["Veronica"]))
    # print(nearestNeighbor("Hailey", users))
    print(recommned('Hailey', users))
