# James Kelly
# Shekh Islam
# NBA award winner prediction

import json, pandas as pd, numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, svm
from sklearn.metrics import classification_report

# Attributes:
# Age, PTS (points), FGM (field goals made), FTM (free throws made),
# AST (assists), STL(steals), BLK(blocks), REB (rebounds)

def main():
    options = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM',
               'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
               'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK',
               'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK',
               'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK',
               'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK',
               'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK', 'CFID', 'CFPARAMS']
    ignore_list = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'W_PCT', 'MIN',
               'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV',
               'BLKA', 'PF', 'PFD', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK',
               'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK',
               'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK',
               'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK',
               'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK', 'CFID', 'CFPARAMS']


    # Prepare Training Data
    # Creates an array of arrays
    # Every inner array is each player's data
    training_all = []
    current_year = 10
    target_year = 19
    trained_results = []
    while (current_year < target_year):
        prev_year = (current_year - 1)
        if prev_year < 10:
            prev_year = "0" + str(prev_year)
        else:
            prev_year = str(prev_year)
        current_file = "leaguedashplayerstats" + str(prev_year) + "-" + str(current_year) + ".json"
        parsed_file = json.load(open(current_file))
        player_entry_count = len(parsed_file['resultSets'][0]['rowSet'])
        player_iter = 0
        while player_iter < player_entry_count:
            player_check = parsed_file['resultSets'][0]['rowSet'][player_iter].copy()
            award_winner = isAwardWinner("20"+str(current_year),player_check[get_param_index('PLAYER_NAME',options)])
            trained_results.append(award_winner)
            training_all.append(player_check)
            player_iter += 1
        current_year += 1


    # Prepare Testing Data and Actual Classifications
    testing_all = []
    test_actual_results = []
    prev_year = 18
    current_year = 19
    current_file = "leaguedashplayerstats" + str(prev_year) + "-" + str(current_year) + ".json"
    parsed_file = json.load(open(current_file))
    player_entry_count = len(parsed_file['resultSets'][0]['rowSet'])
    player_iter = 0
    while player_iter < player_entry_count:
        player_check = parsed_file['resultSets'][0]['rowSet'][player_iter].copy()
        award_winner = isAwardWinner("20" + str(current_year), player_check[get_param_index('PLAYER_NAME', options)])
        test_actual_results.append(award_winner)
        testing_all.append(player_check)
        player_iter += 1


    # Changed listss to Pandas data structure
    pd_train = (pd.DataFrame(training_all, columns=options))
    pd_test = (pd.DataFrame(testing_all, columns=options))
    pd_train_actual = (pd.DataFrame(trained_results, columns=['Winner']))
    pd_test_actual = (pd.DataFrame(test_actual_results, columns=['Winner']))

    # Remove unwanted columns
    pd_train = pd_train.drop(columns=ignore_list)
    pd_test = pd_test.drop(columns=ignore_list)

    # Naive Bayes Model

    NBmodel = MultinomialNB().fit(pd_train, pd_train_actual.values.ravel())
    predicted_nb = NBmodel.predict(pd_test)
    pd_test_predicted_nb = (pd.DataFrame(predicted_nb, columns=['Winner']))

    # Output Results
    print("NAIVE BAYES CLASSIFICATION")
    print("\nPrediction Accuracy for 2019 NBA MVPs")
    print(metrics.accuracy_score(pd_test_actual, pd_test_predicted_nb))
    print("\nConfusion Matrix")
    print(metrics.confusion_matrix(pd_test_actual, pd_test_predicted_nb))
    print("\nClassification Report")
    print(classification_report(pd_test_actual, pd_test_predicted_nb))

    print("\nPotential MVPs for NBA 2019 (NB)")
    iter = 0
    while iter < len(predicted_nb):
        actual = " "
        if bool(predicted_nb[iter]) is True and bool(test_actual_results[iter]) is True:
            actual = " : TP"
        elif bool(predicted_nb[iter]) is True and bool(test_actual_results[iter]) is False:
            actual = " : FP"
        elif bool(predicted_nb[iter]) is False and bool(test_actual_results[iter]) is True:
            actual = " : FN"
        if actual != " ":
            print(testing_all[iter][get_param_index('PLAYER_NAME', options)] + actual)

        iter += 1

    # SVM Model

    SVMmodel = svm.SVC(gamma='scale', kernel='linear').fit(pd_train, pd_train_actual.values.ravel())
    predicted_svm = SVMmodel.predict(pd_test)
    pd_test_predicted_svm = (pd.DataFrame(predicted_svm, columns=['Winner']))

    # Output Results
    print("SVM CLASSIFICATION")
    print("\nPrediction Accuracy for 2019 NBA MVPs")
    print(metrics.accuracy_score(pd_test_actual, pd_test_predicted_svm))
    print("\nConfusion Matrix")
    print(metrics.confusion_matrix(pd_test_actual, pd_test_predicted_svm))
    print("\nClassification Report")
    print(classification_report(pd_test_actual, pd_test_predicted_svm))
    print("\nPotential MVPs for NBA 2019 (SVM)")
    iter = 0
    while iter < len(predicted_nb):
        actual = " "
        if bool(predicted_svm[iter]) is True and bool(test_actual_results[iter]) is True:
            actual = " : TP"
        elif bool(predicted_svm[iter]) is True and bool(test_actual_results[iter]) is False:
            actual = " : FP"
        elif bool(predicted_svm[iter]) is False and bool(test_actual_results[iter]) is True:
            actual = " : FN"
        if actual != " ":
            print(testing_all[iter][get_param_index('PLAYER_NAME', options)] + actual)

        iter += 1



    return 0

def get_param_index(string, options):
    if string in options:
        return options.index(string)
    else:
        return -1

def isAwardWinner(year_string, player_name_string):
    # Using MVP top 5 players
    w2019 = ["Giannis Antetokounmpo", "Paul George", "James Harden", "Nikola Jokic", "Stephen Curry"]
    w2018 = ["James Harden", "LeBron James", "Anthony Davis", "Damian Lillard", "Russel Westbrook"]
    w2017 = ["Russel Westbrook", "James Harden", "Kawhi Leonard", "LeBron James", "Isaiah Thomas"]
    w2016 = ["Stephen Curry", "Kawhi Leonard", "LeBron James", "Russel Westbrook", "Kevin Durant"]
    w2015 = ["Stephen Curry", "James Harden", "LeBron James" ,"Russel Westbrook", "Anthony Davis"]
    w2014 = ["Kevin Durant", "LeBron James", "Blake Griffin", "Joakim Noah", "James Harden"]
    w2013 = ["LeBron James", "Kevin Durant", "Carmelo Anthony", "Chris Paul", "Kobe Bryant"]
    w2012 = ["LeBron James", "Kevin Durant", "Chris Paul", "Kobe Bryant", "Tony Parker"]
    w2011 = ["Derrick Rose", "Dwight Howard", "LeBron James", "Kobe Bryant", "Kevin Durant"]
    w2010 = ["LeBron James", "Kevin Durant", "Kobe Bryant", "Dwight Howard", "Dwayne Wade"]
    if year_string == "2019":
        return player_name_string in w2019
    elif year_string == "2018":
        return player_name_string in w2018
    elif year_string == "2017":
        return player_name_string in w2017
    elif year_string == "2016":
        return player_name_string in w2016
    elif year_string == "2015":
        return player_name_string in w2015
    elif year_string == "2014":
        return player_name_string in w2014
    elif year_string == "2013":
        return player_name_string in w2013
    if year_string == "2012":
        return player_name_string in w2012
    elif year_string == "2011":
        return player_name_string in w2011
    elif year_string == "2010":
        return player_name_string in w2010
    else:
        return False

main()

