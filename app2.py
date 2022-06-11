import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, time
from PIL import Image
from statsbombpy import sb
from scipy.ndimage import gaussian_filter
from mplsoccer.pitch import Pitch, VerticalPitch
pd.options.mode.chained_assignment = None
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import base64
from fpdf import FPDF
from tempfile import NamedTemporaryFile
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
        
#from App import teamEventData



def app():

        image = Image.open('playerPage.jpg')
        
        col1, col2 = st.columns(2)
        
        col1.title("Player Shot Analysis")
        
        col2.image(image)
        
        #download the competition and season data from statsbomb
        df = sb.matches(competition_id=37, season_id=4)
        Teams = df['home_team'].sort_values(ascending=True)
        Teams = Teams.unique()
        

        
        #select the team for analysis
        teamSelect = st.selectbox(
            "Chose the Team", Teams)
        
             
        teamdf = df[(df.home_team == teamSelect ) | (df.away_team == teamSelect)]
        team_match_id = teamdf["match_id"]
        
        @st.cache(allow_output_mutation  = True, suppress_st_warning=True)
        #download the event data for all matches of the season for the seleted team
        def teamEventData(t_match_id):
        
            
            appended_event = []
            st_time = time.time()
            
            for i in t_match_id:
                event = sb.events(match_id = i)
                appended_event.append(event)
            appended_event = pd.concat(appended_event)
            
            et_time = time.time()
            elapsed_time = et_time - st_time
            e_time = str(round(elapsed_time,2))
            st.info(f"data import success in {e_time} seconds " )
            
            return appended_event
        
        final_event_data = teamEventData(team_match_id)
        
        ##PLAYER DATA IMPORT
        
        df_player_data = pd.read_csv('playerdata.csv')
        df_player = df_player_data
        
        df_player= df_player.drop(['playerid','Rk'],axis=1)
        df_team_mins = df_player[df_player["Squad"]== teamSelect]
        
        df_team_mins['player'] = df_team_mins['Player']
        #df_team_mins.rename(columns = {'Player':'player'}, inplace = True)
        df_team_m = df_team_mins
        #df_team_m
        
        df_team_m['player'] = df_team_mins['player'].str.slice(-3)
        #df_team_m
        
        
        ##SHOT DATA PREPARATION
        
        shot_df = final_event_data.loc[(final_event_data['type'] == 'Shot' ) & (final_event_data['team'] == teamSelect)]
        
        shot_df_split_new = shot_df
        
        
        #####MODELLING########
        def xg_model  (shot_df):
        
            #creating new data to use further
            shot_df_split_new = shot_df
            shot_df_split_export = shot_df_split_new
            shot_df_split_new_ex = shot_df_split_new
            ##PREFERENCE FOOT CODE
    
            pref_foot = shot_df_split_new[['player','shot_body_part']].copy()
            
            pref = pref_foot.groupby(["player",'shot_body_part'])["shot_body_part"].count()
            abc =pref.unstack(level=1)
            
            abc = abc.reset_index(level=0)
            abc = abc.replace(np.nan, 0)
            
            abc["player_pref_foot"] = 0
            for i in range(len(abc)) :
                if float(abc.iloc[i,2]) >  float(abc.iloc[i,3]) :
                    abc.iloc[i,4] = 'Left'
                else:
                    abc.iloc[i,4] = 'Right'
            
                    
            pref_foot_player = abc[['player','player_pref_foot']].copy()
            shot_df_split_ex=shot_df_split_new_ex.merge(pref_foot_player,left_on="player",right_on="player")
            
            player_pref_foot = shot_df_split_ex["player_pref_foot"]
            
            abc = player_pref_foot.to_frame()
            
            shot_df_split_new["player_pref_foot"] = 0
            for i in range(len(shot_df_split_new)) :
                shot_df_split_new.iloc[i,-1] =  abc.iloc[i,0]
            
            
            shot_df_split_new[['x', 'y']] = shot_df['location'].apply(pd.Series)
            shot_df_split_new['y'] = 80-shot_df_split_new['y']
            shot_df_split_new[['endx', 'endy', 'endz']] = shot_df['shot_end_location'].apply(pd.Series)
            shot_df_split_new['endy'] = 80-shot_df_split_new['endy']
    
            shot_df_split_new['distance'] =  ((120 - shot_df_split_new['x'] )**2 + (40 - shot_df_split_new['y'] )**2)**0.5
            #shot_df_split_new
            
            for i in range(len(shot_df_split_new)) :
                if shot_df_split_new.iloc[i,-1] ==0 and shot_df_split_new.iloc[i,-1] <=5.5  :
                    shot_df_split_new.iloc[i,-1] = 'Close Range'
                elif shot_df_split_new.iloc[i,-1] > 5.5 and shot_df_split_new.iloc[i,-1] <=16.5  :
                    shot_df_split_new.iloc[i,-1] = 'Penalty Box'
                elif shot_df_split_new.iloc[i,-1] > 16.5 and shot_df_split_new.iloc[i,-1] <=21  :
                    shot_df_split_new.iloc[i,-1] = 'Outside Box'
                elif shot_df_split_new.iloc[i,-1] > 21 and shot_df_split_new.iloc[i,-1] <=32  :
                    shot_df_split_new.iloc[i,-1] = 'Long Range'
                else:
                    shot_df_split_new.iloc[i,-1] ='more_35yd'

    
                        
            shotDistance = {'Close Range':4, 
                'Penalty Box':3, 
                'Outside Box':2, 
                'Long Range':1, 
                'more_35yd':0.5}
            shot_df_split_new['distance'] = shot_df_split_new.distance.map(shotDistance)
            
            
            #FOR END location DISTANCE
            #for i in range(len(shot_df_split_new)) :
            #    if shot_df_split_new.iloc[i,114] ==0 and shot_df_split_new.iloc[i,113] <=5.5  :
            #        shot_df_split_new.iloc[i,114] = 'Very Close'
            #    elif shot_df_split_new.iloc[i,114] > 5.5 and shot_df_split_new.iloc[i,113] <=16.5  :
            #        shot_df_split_new.iloc[i,114] = ''
            #    elif shot_df_split_new.iloc[i,114] > 16.5 and shot_df_split_new.iloc[i,113] <=21  :
            #        shot_df_split_new.iloc[i,114] = 'Outside Box'
            #    elif shot_df_split_new.iloc[i,114] > 21 and shot_df_split_new.iloc[i,113] <=32  :
            #        shot_df_split_new.iloc[i,114] = 'Long Range'
            #    else:
            #        shot_df_split_new.iloc[i,114] ='more_35yd'
    
            
            
            shot_df_split_new.loc[(shot_df_split_new.shot_first_time) == True, 'shot_first_time'] = 1
            shot_df_split_new.loc[(shot_df_split_new.shot_first_time).isna(), 'shot_first_time'] = 0
                    
            shot_df_split_new.loc[(shot_df_split_new.shot_one_on_one) == True, 'shot_one_on_one'] = 1
            shot_df_split_new.loc[(shot_df_split_new.shot_one_on_one).isna(), 'shot_one_on_one'] = 0

            shot_df_split_new.loc[(shot_df_split_new.under_pressure) == True, 'under_pressure'] = 1
            shot_df_split_new.loc[(shot_df_split_new.under_pressure).isna(), 'under_pressure'] = 0
            
            shot_df_split_new.loc[(shot_df_split_new.shot_open_goal) == True, 'shot_open_goal'] = 1
            shot_df_split_new.loc[(shot_df_split_new.shot_open_goal).isna(), 'shot_open_goal'] = 0
            
            num = shot_df_split_new.columns.get_loc("shot_outcome")
            shot_df_split_new['outcome_shot'] = 0
                        
            for i in range(len(shot_df_split_new)) :
                #print(shot_df_split_new.iloc[i,num])
                if shot_df_split_new.iloc[i,num] == 'Goal':
                    shot_df_split_new.iloc[i,-1] = 'On Target'
                elif shot_df_split_new.iloc[i,num] == 'Saved':
                    shot_df_split_new.iloc[i,-1] = 'On Target'
                elif shot_df_split_new.iloc[i,num] == 'Off T':
                    shot_df_split_new.iloc[i,-1] = 'Off Target'
                elif shot_df_split_new.iloc[i,num] == 'Wayward':
                    shot_df_split_new.iloc[i,-1] = 'Off Target'
                elif shot_df_split_new.iloc[i,num] == 'Post':
                    shot_df_split_new.iloc[i,-1] = 'Post'
                else:
                    shot_df_split_new.iloc[i,-1] = 'Blocked'
                    
            shotOutcome = {'Off Target':0, 
            'Blocked':.2, 
            'Post':.4, 
            'On Target':.7}
            shot_df_split_new['outcome_shot'] = shot_df_split_new.outcome_shot.map(shotOutcome)
            
            
            dataraw = shot_df_split_new[['distance','player_pref_foot','shot_body_part','shot_type','shot_technique','outcome_shot','shot_first_time','shot_one_on_one','under_pressure','shot_open_goal']].copy()
            #dataraw = shot_df_split_new[['distance','player_pref_foot','shot_body_part','shot_type','shot_technique','shot_first_time','shot_one_on_one','under_pressure','shot_open_goal']].copy()
            #dataraw        
            
            #data = pd.get_dummies(dataraw, columns=['distance','player_pref_foot', 'shot_body_part','shot_type', 'shot_technique'])
            data = pd.get_dummies(dataraw, columns=['player_pref_foot', 'shot_body_part','shot_type', 'shot_technique'])
            #data
            
            
            num = shot_df_split_new.columns.get_loc("shot_outcome")
            shot_df_split_new['is_goal'] = 0
            
            for i in range(len(shot_df_split_new)) :
                if shot_df_split_new.iloc[i,num] == 'Goal':
                    shot_df_split_new.iloc[i,-1] =1
                else:
                    shot_df_split_new.iloc[i,-1] =0
                        
            data['is_goal'] = shot_df_split_new['is_goal']
            #data
            datatest = data
            
            #train_test_split
    
            X = datatest.iloc[:,:-1]
            y = datatest.iloc[:,-1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)
            
            #SMOTE
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
            #X_train
            
            #from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            st.write('The test set contains {} examples (shots) of which {} are positive (goals).'.format(len(y_test), y_test.sum()))
            st.write('The accuracy of classifying whether a shot is goal or not is {}%.'.format(round(model.score(X_test, y_test)*100),2))
            
            
            #from sklearn.metrics import confusion_matrix
            #print(confusion_matrix(y_test,model.predict(X_test)))
            #
            #from sklearn.metrics import classification_report
            #print(classification_report(y_test,model.predict(X_test)))
    
    
            prob = model.predict_proba(X)
            
            data['xg'] = prob.tolist() 
            
            data[['xg_0', 'xg_1']] = data['xg'].apply(pd.Series)
            
            shot_df_split_new['xg_0'] = data['xg_0']
            shot_df_split_new['xg_1'] = data['xg_1']
            #shot_df_split_export['distance'] = data['distance']
            
            shot_df_split_out = shot_df_split_new[['player','location','shot_end_location','player_pref_foot','distance','shot_outcome','is_goal','shot_statsbomb_xg','xg_1']].copy()
            
            AgGrid(shot_df_split_out.head(20))
            
            return shot_df_split_export
        #####################33
        
             
        
        
        run_model = st.button("Run xG Model")
        
        try:
            if run_model:
                xg_df = xg_model(shot_df)
        except IndexError:
                st.info(f"Model not available for {teamSelect} data set hence using Statbomb xG.")        
            
        except ValueError:
                st.info("Model not available for {teamSelect} data set hence using Statbomb xG.")        
        
        except KeyError:
                st.info("Model not available for {teamSelect} data set hence using Statbomb xG.")        
        
        
        goal_df = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' )]
        #goal_df
        for i in range(len(goal_df)) :
            if goal_df.iloc[i,68] == 'Goal' :
                goal_df['Goal'] =1
            else:
                goal_df['Goal'] =0
        
        totalGoals = len(goal_df)
        
        goal_df_player = goal_df[['player','Goal']].copy()
        
        goal_df_player = goal_df_player.groupby(["player",'Goal'])["Goal"].count()
        goal_df_player = goal_df_player.to_frame()
        abc = goal_df_player.reset_index(level=0)
        goal_df_player = abc.reset_index(drop=True)
        goal_df_player = goal_df_player.sort_values("Goal", ascending=False)
        #goal_df_player
        goal_df_player_m = goal_df_player
        goal_df_player_m['player'] = goal_df_player['player'].str.slice(-3)
        #goal_df_player_m
        
        
        
        #XG DF
        

        xg_df_player = shot_df_split_new[['player','shot_statsbomb_xg']].copy()
        xg_df_player = xg_df_player.groupby(["player"])["shot_statsbomb_xg"].sum()
        xg_df_player = xg_df_player.to_frame()  ####
        ab = xg_df_player.reset_index(level=0)
        xg_df_player = ab.reset_index(drop=True)
        
        xg_df_player = xg_df_player.sort_values("shot_statsbomb_xg", ascending=False)
        #goal_df_player
        xg_df_player_m = xg_df_player
        xg_df_player_m['player'] = xg_df_player['player'].str.slice(-3)
        #xg_df_player_m
        
        mergedPlayer_df = pd.merge(goal_df_player_m, df_team_m,on="player")
        #mergedPlayer_df
        mergedPlayer_df['G_90'] = mergedPlayer_df.Goal/mergedPlayer_df['90s']
        
        mergedPlayer_finaldf = pd.merge(mergedPlayer_df, xg_df_player_m,on="player")
        
        mergedPlayer_finaldf['xg_90'] = mergedPlayer_finaldf.shot_statsbomb_xg/mergedPlayer_df['90s']
        
        mergedPlayer_finaldf.rename(columns = {'shot_statsbomb_xg':'xG'}, inplace = True)
        
        #g_p90 = mergedPlayer_finaldf["G_90"].values
        #xg_p90 = mergedPlayer_finaldf["xg_90"].values
        #
        #init_notebook_mode(connected=True)
        #cf.go_offline()
        #
        #fig = mergedPlayer_finaldf.iplot(kind='scatter',x='G_90',y='xg_90', mode='markers',text='Player',size=10, xTitle='Goal per 90',yTitle='xg per 90',title='Goal vs XG p90')
        #
        #py.plotly_chart(fig)
        
       
        
       ###########################################################################
       
        #select the player for analysis
        players = shot_df_split_new["player"].unique()
        
        #playerSelect = st.selectbox(
        #    "Chose the Player",
        #    players        )

        #st.write(f"You selected {playerSelect}")
        
        
        player_choice = st.multiselect(
    "Choose Players:", players, default=players[0])
        
        st.write(f"Numbers of players selected {len(player_choice)}")
        st.write(f"You selected {player_choice}")
        #shots from single player
        
        
        
        shot_df_split_new[['x', 'y']] = shot_df['location'].apply(pd.Series)
        shot_df_split_new['y'] = 80-shot_df_split_new['y']
        shot_df_split_new[['endx', 'endy', 'endz']] = shot_df['shot_end_location'].apply(pd.Series)
        shot_df_split_new['endy'] = 80-shot_df_split_new['endy']
    
                
        if len(player_choice) == 1:
            figs = []
        
            player1 = player_choice[0]
            
            shot_df_split_pl1 = shot_df_split_new.loc[(shot_df_split_new['player'] == player1)]
    
            shot_df_pl_left1 = shot_df_split_pl1.loc[(shot_df_split_pl1['shot_body_part'] == 'Left Foot')]
            shot_df_pl_right1 = shot_df_split_pl1.loc[(shot_df_split_pl1['shot_body_part'] == 'Right Foot')]
            shot_df_pl_head1 = shot_df_split_pl1.loc[(shot_df_split_pl1['shot_body_part'] == 'Head')]
            
            col3,col4 = st.columns(2)
            
            pitch1 = VerticalPitch(half = True, pitch_type='statsbomb', pitch_color='white', line_color='#c7d5cc' )
            fig1, ax1 = pitch1.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            #fig.set_facecolor('#22312b')
            
            pitch1.scatter(shot_df_pl_left1.x, shot_df_pl_left1.y,marker ='^', color='black', alpha=0.9,edgecolor='black', ax=ax1, s =60, label = 'Left Footed')
            pitch1.scatter(shot_df_pl_right1.x, shot_df_pl_right1.y,marker ='P', color='red', alpha=0.9, ax=ax1, s =60, label = 'Right Footed')
            pitch1.scatter(shot_df_pl_head1.x, shot_df_pl_head1.y,marker ='d', color='blue', alpha=0.9,ax=ax1, s =60, label = 'Header')
            ax1.legend(loc='lower right').get_texts()[0].set_color("black")
            plt.title(f"{player1}'s Shot Map ", fontsize=16, fontfamily='serif')
            col3.pyplot(fig1) 
            figs.append(fig1)
            
            hmPitch1 = VerticalPitch(half = True, pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
            hmFig1, hmAxs1 = hmPitch1.grid(figheight=9, title_height=0.05, endnote_space=0, axis=False, title_space=0, grid_height=0.8, endnote_height=0.05)
            hmFig1.set_facecolor('white')

            bin_statistic1 = hmPitch1.bin_statistic(shot_df_split_pl1.x, shot_df_split_pl1.y, statistic='count', bins=(20, 20)) 
            bin_statistic1['statistic'] = gaussian_filter(bin_statistic1['statistic'], 1)
            pcm1 = hmPitch1.heatmap(bin_statistic1, ax=hmAxs1['pitch'], cmap='hot', edgecolors='#22312b')
            cbar1 = hmFig1.colorbar(pcm1, ax=hmAxs1['pitch'], shrink=0.6)
            cbar1.outline.set_edgecolor('#efefef')
            plt.title(f"{player1}'s Shot Density Heat Map", fontsize=16, fontfamily='serif')
            col4.pyplot(hmFig1)  
            figs.append(hmFig1)
            
            ##SHOT OUTCOME
            playershots1 = shot_df_split_pl1[shot_df_split_pl1.type=='Shot']
            playerPie1 = playershots1[['shot_outcome', 'id']].groupby('shot_outcome').count().reset_index().rename(columns={'id': 'count'})
            
            figSO1, ax1 = plt.subplots(figsize=[8,8])
            labels1 = playerPie1['shot_outcome']
            lenX = len(playerPie1['count'])
            colors1 = ['#ff9999','#66b3ff','#99ff99','#ffcc99','purple', 'red', 'orange', 'yellow']
            plt.pie(x=playerPie1['count'], autopct="%.1f%%", explode=[0.01]*lenX, pctdistance=0.7, colors=colors1, \
            textprops=dict(fontsize=16))
            plt.legend(labels1,loc="center left",bbox_to_anchor=(1,0, 5, 1))
            plt.title("Shot Outcomes", fontsize=16, fontfamily='serif')
            plt.tight_layout()
            
            col5,col6 = st.columns(2)
            col5.pyplot(figSO1)
            figs.append(figSO1)
            
            
            goal_df = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' ) ]

        ########################### STACK BAR GOAL XG################
        
            num1 = shot_df_split_new.columns.get_loc("shot_outcome")
            
            
            #total goals scored by time 
            for i in range(len(goal_df)) :
                if goal_df.iloc[i,num1] == 'Goal' :
                            goal_df['Goal'] =1
                else:
                    goal_df['Goal'] =0        
            total_team_goal = len(goal_df)
            #total_team_goal
            
            
            #Total goals scored by player 
            goal_df_player = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' ) & (shot_df_split_new['player'] == player1)]
            
            
            for i in range(len(goal_df)) :
                if goal_df.iloc[i,num1] == 'Goal' :
                            goal_df['Goal'] =1
                else:
                    goal_df['Goal'] =0        
            
            total_goal_player = len(goal_df_player)
            #total_goal_player
            
            #total shots by player 
            player_shot_df = shot_df_split_new[(shot_df_split_new['type'] == 'Shot' ) & (shot_df_split_new['player'] == player1)]
            
            
            for i in range(len(player_shot_df)) :
                if player_shot_df.iloc[i,num1] == 'Shot' :
                            goal_df['Shot'] =1
                else:
                    player_shot_df['Shot'] =0        
            total_shots_player = len(player_shot_df)
            #total_shots_player
            
            #total shots by team 
            total_shots = len(shot_df)
            #total_shots 
            
            #xg player
            xg_df = shot_df_split_new[(shot_df_split_new['player'] == player1)]
            
            xg_player = xg_df['shot_statsbomb_xg'].sum()
                        
            #xg team #
            xg_total = shot_df_split_new['shot_statsbomb_xg'].sum()
            
            penalty_df = shot_df_split_new[(shot_df_split_new['shot_type'] == 'Penalty' )] #& (shot_df_split_new['player'] == player1)]
            
            num2 = shot_df_split_new.columns.get_loc("shot_type")
            
            for i in range(len(penalty_df)) :
                if penalty_df.iloc[i,num2] == 'Penalty' :
                            penalty_df['Penalty'] =1
                else:
                    penalty_df['Penalty'] =0        
            
            
            penalty_df_player = shot_df_split_new[(shot_df_split_new['shot_type'] == 'Penalty' ) & (shot_df_split_new['player'] == player1)]
            
            
            for i in range(len(penalty_df_player)) :
                if penalty_df_player.iloc[i,num2] == 'Penalty' :
                            penalty_df_player['Penalty'] =1
                else:
                    penalty_df_player['Penalty'] =0
            
            total_penalty = len(penalty_df)
            
            penalty_xg = total_penalty*0.76
            
            nonpenalty_total = xg_total - penalty_xg 
            
            penalty_xg_player = len(penalty_df_player)*0.76
            
            non_penalty_xg_player = xg_player - penalty_xg_player
           
            # create data
            fig_stk, ax = plt.subplots(figsize=[8,8])
            prop_df = pd.DataFrame([['Goals', total_goal_player/total_team_goal*100, (total_team_goal/total_team_goal*100-total_goal_player/total_team_goal*100)], ['Shots', total_shots_player/total_shots*100, (total_shots/total_shots*100-total_shots_player/total_shots*100)], ['XG', xg_player/xg_total*100, (xg_total/xg_total*100 - xg_player/xg_total*100)],
                            ['NPXG', non_penalty_xg_player/nonpenalty_total*100, (nonpenalty_total/nonpenalty_total*100 - non_penalty_xg_player/nonpenalty_total*100)]],
                            columns=['Proportion', 'Player','Team' ])
            
                
            ax = prop_df.plot(color= ['#24b1d1', '#ae24d1'], x='Proportion', kind='barh', stacked=True, figsize=(10, 6),title=f'{player1} Contribution', ax=ax)
                    
            col6.pyplot(fig_stk)
            figs.append(fig_stk)
            
            
            #######xg vs g per match
            
            goal_df = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' ) & (shot_df_split_new['player'] == player1)]

            num1 = shot_df_split_new.columns.get_loc("shot_outcome")
            
            for i in range(len(goal_df)) :
                if goal_df.iloc[i,num1] == 'Goal' :
                            goal_df['Goal'] =1
                else:
                    goal_df['Goal'] =0  
                    
            if 'Goal' in goal_df.columns:
                actual_goals = goal_df.groupby(['match_id'])['Goal'].sum()
                actual_goals.to_frame()
                actual_goals = actual_goals.to_frame().reset_index()
                
                Expected_goals = xg_df.groupby(['match_id'])['shot_statsbomb_xg'].sum()
                Expected_goals.to_frame()
                Expected_goals = Expected_goals.to_frame().reset_index()
                
                merged_goal_df = pd.merge(Expected_goals, actual_goals,on="match_id",  how='left')
                merged_df = merged_goal_df.fillna(0)
                merged_df['g_c'] = 0
                
                s = 0
                for i in range(len(merged_df)):
                    s+= merged_df.iloc[i,2]
                    merged_df.iloc[i,3] = s
                
                
                merged_df['xg_c'] = 0
                s = 0
                for i in range(len(merged_df)):
                    s+= merged_df.iloc[i,1]
                    merged_df.iloc[i,4] = s
                
                actual_goals_c = merged_df.groupby(['match_id'])['g_c'].sum()
                Expected_goals_c = merged_df.groupby(['match_id'])['xg_c'].sum()
                
                
                summary = pd.concat([actual_goals_c, Expected_goals_c], axis=1)
                
                fig_trend, axis = plt.subplots(figsize=[8,8])           
                axis = summary.plot(ax=axis)
                axis.set_xlabel('Match ID')
                axis.set_ylabel('Expected goals')
                axis.set_title('Expected Goals VS Actual Goals Per match')
                
                cols, colt = st.columns(2)
                
                #cols.pyplot(fig_trend)
                cols.line_chart(summary)
                figs.append(axis)
            else:
                st.info("No goals for the player, Hence no Expected Goals VS Actual Goals Per match graph available")
            
            
            export_as_pdf = st.button("Export Visualisations as Report")
            
            def create_download_link(val, filename):
                b64 = base64.b64encode(val)  # val looks like b'...'
                return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
        
            

            if export_as_pdf:
                pdf = FPDF()
                for fig in figs:
                    pdf.add_page()
                    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name)
                        pdf.image(tmpfile.name, 10, 10, 200, 100)
                html = create_download_link(pdf.output(dest="S").encode("latin-1"), "report")
                st.markdown(html, unsafe_allow_html=True)
            
            
        
        elif len(player_choice) > 2:
        
            st.error("Please select max 2 players for comparision")
        
        elif len(player_choice) == 0 :
            st.error("Please select Atleast 1 player")
            
        else:
            player1 = player_choice[0]
            player2 = player_choice[1]
            
            figs = []
            
            shot_df_split_pl1 = shot_df_split_new.loc[(shot_df_split_new['player'] == player1)]
    
            shot_df_pl_left1 = shot_df_split_pl1.loc[(shot_df_split_pl1['shot_body_part'] == 'Left Foot')]
            shot_df_pl_right1 = shot_df_split_pl1.loc[(shot_df_split_pl1['shot_body_part'] == 'Right Foot')]
            shot_df_pl_head1 = shot_df_split_pl1.loc[(shot_df_split_pl1['shot_body_part'] == 'Head')]
    
            #leftShots = len(shot_df_pl_left)
            #rightShots = len(shot_df_pl_right)
            #headers = len(shot_df_pl_head)
            
            #st.write(f"Number of left footed shots : {leftShots} ")
            #st.write(f"Number of right footed shots : {rightShots} ")
            #st.write(f"Number of headers  : {headers} ")
            
            
            shot_df_split_pl2 = shot_df_split_new.loc[(shot_df_split_new['player'] == player2)]
    
            shot_df_pl_left2 = shot_df_split_pl2.loc[(shot_df_split_pl2['shot_body_part'] == 'Left Foot')]
            shot_df_pl_right2 = shot_df_split_pl2.loc[(shot_df_split_pl2['shot_body_part'] == 'Right Foot')]
            shot_df_pl_head2 = shot_df_split_pl2.loc[(shot_df_split_pl2['shot_body_part'] == 'Head')]
    
            #leftShots = len(shot_df_pl_left)
            #rightShots = len(shot_df_pl_right)
            #headers = len(shot_df_pl_head)
            
            col3, col4 = st.columns(2)
            
            pitch1 = VerticalPitch(half = True, pitch_type='statsbomb', pitch_color='white', line_color='#c7d5cc' )
            fig1, ax1 = pitch1.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            #fig.set_facecolor('#22312b')
            
            pitch1.scatter(shot_df_pl_left1.x, shot_df_pl_left1.y,marker ='^', color='black', alpha=0.9,edgecolor='black', ax=ax1, s =60, label = 'Left Footed')
            pitch1.scatter(shot_df_pl_right1.x, shot_df_pl_right1.y,marker ='P', color='red', alpha=0.9, ax=ax1, s =60, label = 'Right Footed')
            pitch1.scatter(shot_df_pl_head1.x, shot_df_pl_head1.y,marker ='d', color='blue', alpha=0.9,ax=ax1, s =60, label = 'Header')
            ax1.legend(loc='lower right').get_texts()[0].set_color("black")
            plt.title(f"{player1}'s Shot Map ")
            col3.pyplot(fig1) 
            figs.append(fig1)
            
            
            ##PLAYER 2
            pitch2 = VerticalPitch(half = True, pitch_type='statsbomb', pitch_color='white', line_color='#c7d5cc' )
            fig2, ax2 = pitch2.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            #fig.set_facecolor('#22312b')
              
            pitch2.scatter(shot_df_pl_left2.x, shot_df_pl_left2.y,marker ='^', color='black', alpha=0.9,edgecolor='black', ax=ax2, s =60, label = 'Left Footed')
            pitch2.scatter(shot_df_pl_right2.x, shot_df_pl_right2.y,marker ='P', color='red', alpha=0.9, ax=ax2, s =60, label = 'Right Footed')
            pitch2.scatter(shot_df_pl_head2.x, shot_df_pl_head2.y,marker ='d', color='blue', alpha=0.9,ax=ax2, s =60, label = 'Header')
            ax2.legend(loc='lower right').get_texts()[0].set_color("black") 
            plt.title(f"{player2}'s Shot Map ")            
            col4.pyplot(fig2)        
            figs.append(fig2)
               
            col5, col6 = st.columns(2)
            
            ## heatmap 1
            hmPitch1 = VerticalPitch(half = True, pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
            hmFig1, hmAxs1 = hmPitch1.grid(figheight=9, title_height=0.05, endnote_space=0, axis=False, title_space=0, grid_height=0.8, endnote_height=0.05)
            hmFig1.set_facecolor('white')

            bin_statistic1 = hmPitch1.bin_statistic(shot_df_split_pl1.x, shot_df_split_pl1.y, statistic='count', bins=(20, 20)) 
            bin_statistic1['statistic'] = gaussian_filter(bin_statistic1['statistic'], 1)
            pcm1 = hmPitch1.heatmap(bin_statistic1, ax=hmAxs1['pitch'], cmap='hot', edgecolors='#22312b')
            cbar1 = hmFig1.colorbar(pcm1, ax=hmAxs1['pitch'], shrink=0.6)
            cbar1.outline.set_edgecolor('#efefef')
            plt.title(f"{player1}'s Shot Density Heat Map")
            col5.pyplot(hmFig1)
            figs.append(hmFig1)
            
            ## heatmap 2
            hmPitch2 = VerticalPitch(half = True, pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
            hmFig2, hmAxs2 = hmPitch2.grid(figheight=9, title_height=0.05, endnote_space=0, axis=False, title_space=0, grid_height=0.8, endnote_height=0.05)
            hmFig2.set_facecolor('white')

            bin_statistic2 = hmPitch2.bin_statistic(shot_df_split_pl2.x, shot_df_split_pl2.y, statistic='count', bins=(20, 20)) 
            bin_statistic2['statistic'] = gaussian_filter(bin_statistic2['statistic'], 1)
            pcm2 = hmPitch2.heatmap(bin_statistic2, ax=hmAxs2['pitch'], cmap='hot', edgecolors='#22312b')
            cbar2 = hmFig2.colorbar(pcm2, ax=hmAxs2['pitch'], shrink=0.6)
            cbar2.outline.set_edgecolor('#efefef')
            plt.title(f"{player2}'s Shot Density Heat Map")
            col6.pyplot(hmFig2)  
            figs.append(hmFig2)
            
            #individual shot outcome 
            playershots1 = shot_df_split_pl1[shot_df_split_pl1.type=='Shot']
            playerPie1 = playershots1[['shot_outcome', 'id']].groupby('shot_outcome').count().reset_index().rename(columns={'id': 'count'})
            
            figSO1, ax1 = plt.subplots(figsize=[8,8])
            labels1 = playerPie1['shot_outcome']
            lenX = len(playerPie1['count'])
            colors1 = ['#ff9999','#66b3ff','#99ff99','#ffcc99','purple', 'red', 'orange', 'yellow']
            plt.pie(x=playerPie1['count'], autopct="%.1f%%", labels=labels1, explode=[0.04]*lenX, pctdistance=0.7, colors=colors1, shadow=True, \
            textprops=dict(fontsize=16))
            plt.title("Shot Outcomes", fontsize=26, fontfamily='serif')
            plt.tight_layout()
            
            playershots2 = shot_df_split_pl2[shot_df_split_pl2.type=='Shot']
            playerPie2 = playershots2[['shot_outcome', 'id']].groupby('shot_outcome').count().reset_index().rename(columns={'id': 'count'})
            
            figSO2, ax2 = plt.subplots(figsize=[8,8])
            labels2 = playerPie2['shot_outcome']
            lenX = len(playerPie2['count'])
            colors2 = ['#ff9999','#66b3ff','#99ff99','#ffcc99','purple', 'red', 'orange', 'yellow']
            plt.pie(x=playerPie2['count'], autopct="%.1f%%", labels=labels2, explode=[0.04]*lenX, pctdistance=0.7, colors=colors2, shadow=True, \
            textprops=dict(fontsize=16))
            plt.title("Shot Outcomes", fontsize=26, fontfamily='serif')
            plt.tight_layout()
            
            
            col7, col8 = st.columns(2)
            col7.pyplot(figSO1)
            col8.pyplot(figSO2)
            figs.append(figSO1)
            figs.append(figSO2)
            
            ##BAR CHART
            
            num1 = shot_df_split_new.columns.get_loc("shot_outcome")
            
            
            #total goals scored by time 
            for i in range(len(goal_df)) :
                if goal_df.iloc[i,num1] == 'Goal' :
                            goal_df['Goal'] =1
                else:
                    goal_df['Goal'] =0        
            total_team_goal = len(goal_df)
            #total_team_goal
            
            
            #Total goals scored by player 
            goal_df_player1 = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' ) & (shot_df_split_new['player'] == player1)]
            goal_df_player2 = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' ) & (shot_df_split_new['player'] == player1)]
            
            for i in range(len(goal_df)) :
                if goal_df.iloc[i,num1] == 'Goal' :
                            goal_df['Goal'] =1
                else:
                    goal_df['Goal'] =0        
            
            total_goal_player1 = len(goal_df_player1)
            total_goal_player2 = len(goal_df_player2)
            #total_goal_player
            
            #total shots by player 
            player1_shot_df = shot_df_split_new[(shot_df_split_new['type'] == 'Shot' ) & (shot_df_split_new['player'] == player1)]
            player2_shot_df = shot_df_split_new[(shot_df_split_new['type'] == 'Shot' ) & (shot_df_split_new['player'] == player2)]
            
            for i in range(len(player1_shot_df)) :
                if player1_shot_df.iloc[i,num1] == 'Shot' :
                            goal_df['Shot'] =1
                else:
                    player1_shot_df['Shot'] =0        
            total_shots_player1 = len(player1_shot_df)
            #total_shots_player
            
            for i in range(len(player2_shot_df)) :
                if player2_shot_df.iloc[i,num1] == 'Shot' :
                            goal_df['Shot'] =1
                else:
                    player2_shot_df['Shot'] =0        
            total_shots_player2 = len(player2_shot_df)
            #total_shots_player
            
            
            #total shots by team 
            total_shots = len(shot_df)
            #total_shots 
            
            #xg player
            xg_df1 = shot_df_split_new[(shot_df_split_new['player'] == player1)]
            xg_df2 = shot_df_split_new[(shot_df_split_new['player'] == player2)]
            
            xg_player1 = xg_df1['shot_statsbomb_xg'].sum()
            xg_player2 = xg_df2['shot_statsbomb_xg'].sum()
            
            #xg team #
            xg_total = shot_df_split_new['shot_statsbomb_xg'].sum()
            
            penalty_df = shot_df_split_new[(shot_df_split_new['shot_type'] == 'Penalty' )] #& (shot_df_split_new['player'] == player1)]
            
            num2 = shot_df_split_new.columns.get_loc("shot_type")
            
            for i in range(len(penalty_df)) :
                if penalty_df.iloc[i,num2] == 'Penalty' :
                            penalty_df['Penalty'] =1
                else:
                    penalty_df['Penalty'] =0        
            
            
            penalty_df_player1 = shot_df_split_new[(shot_df_split_new['shot_type'] == 'Penalty' ) & (shot_df_split_new['player'] == player1)]
            penalty_df_player2 = shot_df_split_new[(shot_df_split_new['shot_type'] == 'Penalty' ) & (shot_df_split_new['player'] == player2)]
            
            for i in range(len(penalty_df_player1)) :
                if penalty_df_player1.iloc[i,num2] == 'Penalty' :
                            penalty_df_player1['Penalty'] =1
                else:
                    penalty_df_player1['Penalty'] =0
            
            for i in range(len(penalty_df_player2)) :
                if penalty_df_player2.iloc[i,num2] == 'Penalty' :
                            penalty_df_player2['Penalty'] =1
                else:
                    penalty_df_player2['Penalty'] =0
            
            
            total_penalty = len(penalty_df)
            
            penalty_xg = total_penalty*0.76
            
            nonpenalty_total = xg_total - penalty_xg 
            
            penalty_xg_player1 = len(penalty_df_player1)*0.76
            penalty_xg_player2 = len(penalty_df_player2)*0.76
            
            non_penalty_xg_player1 = xg_player1 - penalty_xg_player1
            non_penalty_xg_player2 = xg_player2 - penalty_xg_player2
            
            # create data
            fig_stk1, ax1 = plt.subplots(figsize=[8,8])
            prop_df1 = pd.DataFrame([['Goals', total_goal_player1/total_team_goal*100, (total_team_goal/total_team_goal*100-total_goal_player1/total_team_goal*100)], ['Shots', total_shots_player1/total_shots*100, (total_shots/total_shots*100-total_shots_player1/total_shots*100)], ['XG', xg_player1/xg_total*100, (xg_total/xg_total*100 - xg_player1/xg_total*100)],
                            ['NPXG', non_penalty_xg_player1/nonpenalty_total*100, (nonpenalty_total/nonpenalty_total*100 - non_penalty_xg_player1/nonpenalty_total*100)]],
                            columns=['Proportion', 'Player','Team' ])
            
                
            ax1 = prop_df1.plot(color= ['#24b1d1', '#ae24d1'], x='Proportion', kind='barh', stacked=True, figsize=(10, 6),title=f'{player1} Contribution', ax=ax1)
                    
            
            fig_stk2, ax2 = plt.subplots(figsize=[8,8])
            prop_df2 = pd.DataFrame([['Goals', total_goal_player2/total_team_goal*100, (total_team_goal/total_team_goal*100-total_goal_player2/total_team_goal*100)], ['Shots', total_shots_player2/total_shots*100, (total_shots/total_shots*100-total_shots_player2/total_shots*100)], ['XG', xg_player2/xg_total*100, (xg_total/xg_total*100 - xg_player2/xg_total*100)],
                            ['NPXG', non_penalty_xg_player2/nonpenalty_total*100, (nonpenalty_total/nonpenalty_total*100 - non_penalty_xg_player2/nonpenalty_total*100)]],
                            columns=['Proportion', 'Player','Team' ])
            
                
            ax2 = prop_df2.plot(color= ['#24b1d1', '#ae24d1'], x='Proportion', kind='barh', stacked=True, figsize=(10, 6),title=f'{player2} Contribution', ax=ax2)
            
            
            col9, col10 = st.columns(2)
            col9.pyplot(fig_stk1)
            col10.pyplot(fig_stk2)
            figs.append(fig_stk1)
            figs.append(fig_stk2)
            
            
            #######xg vs g per match
            #
            #def xgTrend(player):
            #
            #    goal_df = shot_df_split_new[(shot_df_split_new['shot_outcome'] == 'Goal' ) & (shot_df_split_new['player'] == player)]
            #
            #    num1 = shot_df_split_new.columns.get_loc("shot_outcome")
            #    
            #    goal_df['Goal'] =0
            #    for i in range(len(goal_df)) :
            #        if goal_df.iloc[i,num1] == 'Goal' :
            #                    goal_df['Goal'] =1
            #        else:
            #            goal_df['Goal'] =0        
            #    
            #    actual_goals = goal_df.groupby(['match_id'])['Goal'].sum()
            #    actual_goals.to_frame()
            #    actual_goals = actual_goals.to_frame().reset_index()
            #    
            #    Expected_goals = goal_df.groupby(['match_id'])['shot_statsbomb_xg'].sum()
            #    Expected_goals.to_frame()
            #    Expected_goals = Expected_goals.to_frame().reset_index()
            #    
            #    merged_goal_df = pd.merge(Expected_goals, actual_goals,on="match_id",  how='left')
            #    merged_df = merged_goal_df.fillna(0)
            #    merged_df['g_c'] = 0
            #    
            #    s = 0
            #    for i in range(len(merged_df)):
            #        s+= merged_df.iloc[i,2]
            #        merged_df.iloc[i,3] = s
            #    
            #    
            #    merged_df['xg_c'] = 0
            #    s = 0
            #    for i in range(len(merged_df)):
            #        s+= merged_df.iloc[i,1]
            #        merged_df.iloc[i,4] = s
            #    
            #    actual_goals_c = merged_df.groupby(['match_id'])['g_c'].sum()
            #    Expected_goals_c = merged_df.groupby(['match_id'])['xg_c'].sum()
            #    
            #    
            #    summary = pd.concat([actual_goals_c, Expected_goals_c], axis=1)
            #    
            #    
            #    return summary
            #
            #summary1 = xgTrend(player1)
            #summary2 = xgTrend(player2)
            #
            #fig_trend1, axis1 = plt.subplots(figsize=[8,8])           
            #axis1 = summary1.plot(ax=axis1)
            #axis1.set_xlabel('Match ID')
            #axis1.set_ylabel('Expected goals')
            #axis1.set_title('Expected Goals VS Actual Goals Per match')
            #figs.append(axis1)
            #    
            #fig_trend2, axis2 = plt.subplots(figsize=[8,8])           
            #axis2 = summary2.plot(ax=axis2)
            #axis2.set_xlabel('Match ID')
            #axis2.set_ylabel('Expected goals')
            #axis2.set_title('Expected Goals VS Actual Goals Per match')
            #figs.append(axis1)
            #
            #
            #cols, colt = st.columns(2)
            #cols.line_chart(summary1)
            #colt.line_chart(summary2)    
            
            
            export_as_pdf = st.button("Export Visualisations as Report")
            
            def create_download_link(val, filename):
                b64 = base64.b64encode(val)  # val looks like b'...'
                return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
        
            

            if export_as_pdf:
                pdf = FPDF()
                for fig in figs:
                    pdf.add_page()
                    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name)
                        pdf.image(tmpfile.name, 10, 10, 200, 100)
                html = create_download_link(pdf.output(dest="S").encode("latin-1"), "report")
                st.markdown(html, unsafe_allow_html=True)
            