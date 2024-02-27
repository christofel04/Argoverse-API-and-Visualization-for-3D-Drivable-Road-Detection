import pandas as pd
import numpy as np

import os
import pickle

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

def convert_lidar_points_to_bev_tensors( lidar_pts , voxel_size = [ 0.2 , 0.2 ] , max_number_pillars = 21000 , max_number_points = 32 , lidar_range = [-50 , 50 , -50 , 70 ]):

    if lidar_range[0] < 0 :
        lidar_range[1] = lidar_range[1] + -1* lidar_range[0]
        lidar_pts[ : , 0 ] = lidar_pts[ : , 0 ] + -1 * lidar_range[0]
        lidar_range[0] = 0        

        

    if lidar_range[2] < 0 :
        lidar_range[3] = lidar_range[3] + -1* lidar_range[2]
        lidar_pts[ : , 1 ] = lidar_pts[ : , 1 ] + -1* lidar_range[2]
        lidar_range[2] = 0       

        

    
    length_of_pillar_bev = int( ( lidar_range[ 1] - lidar_range[0] )/ voxel_size[0] )
    width_of_pillar_bev = int( ( lidar_range[ 3 ] - lidar_range[2] )/ voxel_size[1] )
    
    list_of_points_in_tensor = [[[] for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]

    for lidar_point in lidar_pts :

        X_coordinate_lidar_point = int( lidar_point[0]//voxel_size[0] )

        if (( X_coordinate_lidar_point < 0 ) | ( X_coordinate_lidar_point >= length_of_pillar_bev ))  :
            continue
            
        Y_coordinate_lidar_point = int( lidar_point[1]//voxel_size[1] )

        if (( Y_coordinate_lidar_point < 0 ) | ( Y_coordinate_lidar_point >= width_of_pillar_bev )) :
            continue

        #print( "Select lidar points : " + str( lidar_point ))

        list_of_points_in_tensor[ X_coordinate_lidar_point][ Y_coordinate_lidar_point ].append( lidar_point )
        

    list_of_bev_tensor = []

    list_of_bev_tensor_indices = []
    
    for x in range( length_of_pillar_bev ) :

        for y in range( width_of_pillar_bev ) :

            if len( list_of_points_in_tensor[x][y] ) > 0 :
    
                if len( list_of_points_in_tensor[x][y] ) > max_number_points :
    
                    list_of_selected_index = np.linspace( 0 , len( list_of_points_in_tensor[x][y] ) -1 , max_number_points ).astype(int)

                    #print( "List of selected index : " + str( list_of_selected_index ))
    
                    selected_points_in_pillar = np.array( list_of_points_in_tensor[x][y])[list_of_selected_index]
    
                    # Extract different of points coordinate to all points in pillar and different of points coordinate to all pillar BEV center
    
                    average_coordinate_selected_points_in_pillar = selected_points_in_pillar[ : , : ].mean( axis = 0 )

                    average_coordinate_selected_points_in_pillar[ 3 ] = 0
    
                    different_coordinate_selected_points_in_pillar = selected_points_in_pillar - average_coordinate_selected_points_in_pillar
    
                    center_of_bev_pillar = [ x* voxel_size[0] + voxel_size[0]/2 , y* voxel_size[1] + voxel_size[1]/2 ]
    
                    different_coordinate_selected_points_in_pillar_to_bev_pillar_center = selected_points_in_pillar[ : , : 2 ] - center_of_bev_pillar
    
                    extracted_selected_points_in_pillar = np.concatenate( [ selected_points_in_pillar , different_coordinate_selected_points_in_pillar[ : , :3] , different_coordinate_selected_points_in_pillar_to_bev_pillar_center ] , axis = 1 )
    
                    list_of_bev_tensor.append( extracted_selected_points_in_pillar )
    
                else :
    
                    selected_points_in_pillar = np.array( list_of_points_in_tensor[x][y][ : ] )
    
                    #print( "Selected points in pillar : " + str( selected_points_in_pillar ))
    
                    # Extract different of points coordinate to all points in pillar and different of points coordinate to all pillar BEV center
    
                    average_coordinate_selected_points_in_pillar = selected_points_in_pillar[ : , : ].mean( axis = 0 )

                    #print( "Average coordinate selected points in pillar : " + str( average_coordinate_selected_points_in_pillar ))

                    average_coordinate_selected_points_in_pillar[3] = 0
    
                    different_coordinate_selected_points_in_pillar = selected_points_in_pillar - average_coordinate_selected_points_in_pillar
    
                    center_of_bev_pillar = [ x* voxel_size[0] + voxel_size[0]/2 , y* voxel_size[1] + voxel_size[1]/2 ]
    
                    different_coordinate_selected_points_in_pillar_to_bev_pillar_center = selected_points_in_pillar[ : , : 2 ] - center_of_bev_pillar
    
                    extracted_selected_points_in_pillar = np.concatenate( [ selected_points_in_pillar , different_coordinate_selected_points_in_pillar[ : , :3] , different_coordinate_selected_points_in_pillar_to_bev_pillar_center ] , axis = 1 )
    
                    #for padding_bev_tensor in range( max_number_points - len( list_of_points_in_tensor[x][y] ) ) :
    
                    extracted_selected_points_in_pillar = np.append(extracted_selected_points_in_pillar , np.array( [[ 0 for i in range(9) ] for j in range( max_number_points - len( list_of_points_in_tensor[x][y]))]).reshape( -1,9) , axis = 0)
    
                    #print( "Final extracted points in pillar : " + str( extracted_selected_points_in_pillar ))
                    list_of_bev_tensor.append(extracted_selected_points_in_pillar )
                        
                list_of_bev_tensor_indices.append(np.array([ 0 , x , y ] ))

    list_of_bev_tensor = np.stack( list_of_bev_tensor , axis = 0 )
    list_of_bev_tensor_indices = np.stack( list_of_bev_tensor_indices , axis = 0)

    #print( "List of BEV tensor : " + str( list_of_bev_tensor ))
    #print( "List of BEV tensor indices : " + str( list_of_bev_tensor_indices ))

    print( "Dimension of BEV tensor pillars without additional pillars : " + str( list_of_bev_tensor.shape ))

                                                   
    if list_of_bev_tensor.shape[0] > max_number_pillars :

        selected_pillar_indices = np.linspace( 0 , list_of_bev_tensor.shape[0] , max_number_pillars )

        list_of_bev_tensor = list_of_bev_tensor[ : , selected_pillar_indices ]

        list_of_bev_tensor_indices = list_of_bev_tensor_indices[ : , selected_pillar_indices ]

    else :

        #for padding_bev_tensor_pillar in range( max_number_pillars - list_of_bev_tensor.shape[0] ) :

        number_of_padding_tensor = max_number_pillars - list_of_bev_tensor.shape[0]

        list_of_bev_tensor = np.append( list_of_bev_tensor , np.array( [[[0 for i in range(9)] for j in range( max_number_points ) ] for l in range( number_of_padding_tensor )] ).reshape( number_of_padding_tensor , max_number_points , -1 ) , axis = 0 )

        #padding_list_of_bev_tensor  = np.array( [[0 for i in range(3)] for j in range( max_number_pillars - list_of_bev_tensor.shape[0] )])

        #print( "Dimension of padding list of BEV tensor : " + str( padding_list_of_bev_tensor.shape ))
                                               
        list_of_bev_tensor_indices = np.append( list_of_bev_tensor_indices , np.array( [[0 for i in range(3)] for j in range( number_of_padding_tensor )] ), axis = 0)

    list_of_bev_tensor = list_of_bev_tensor.reshape( 1 , max_number_pillars , max_number_points , 9)
    list_of_bev_tensor_indices = list_of_bev_tensor_indices.reshape( 1 , max_number_pillars , 3 )

    print( "Dimension of BEV tensor : " + str( list_of_bev_tensor.shape ))
    print( "Dimension of BEV tensor Index : " + str( list_of_bev_tensor_indices.shape ))
    
    return( list_of_bev_tensor , list_of_bev_tensor_indices )
            

#def main() :

tracking_train_dataset_dir = '/home/ofel04/Downloads/tracking_train1_v1.1/argoverse-tracking/train1'

argoverse_loader = ArgoverseTrackingLoader( tracking_train_dataset_dir )

max_number_log_extracted = 23 #len( argoverse_loader.log_list )

for index_log_argoverse in range( 0 , max_number_log_extracted  ) :

    log_id = argoverse_loader.log_list[ index_log_argoverse ]

    # Create folder for BEV tensor in the log

    name_of_bev_tensor_folder =  tracking_train_dataset_dir + "/" + str( log_id ) + "/BEV_tensor_folder/"

    os.makedirs( name_of_bev_tensor_folder , exist_ok=True )

    name_of_bev_drivable_area_label_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/BEV_drivable_area_label/"

    os.makedirs( name_of_bev_drivable_area_label_folder , exist_ok= True )

    print( "Making BEV tensor for Log Argoverse Dataset : " + str( index_log_argoverse ))

    argoverse_data = argoverse_loader[ index_log_argoverse ]

    number_samples = len( argoverse_loader._lidar_timestamp_list[ str( log_id ) ] )

    for frame_argoverse_index in range( number_samples ) :

        """
        
        lidar_pts = argoverse_data.get_lidar_in_rasterized_map_coordinate( frame_argoverse_index )

        bev_tensor = convert_lidar_points_to_bev_tensors( lidar_pts )

        #print( "List of Argoverse 1 LiDAR data : " + str( argoverse_data._lidar_list ))

        name_of_lidar_frame_file = name_of_bev_tensor_folder + str( argoverse_data._lidar_list[ log_id ][ frame_argoverse_index ] ).split( "/")[-1].replace( ".ply" , "" ) + ".pickle"

        with open( name_of_lidar_frame_file , 'wb+') as handle:
            pickle.dump( bev_tensor, handle)

        print( "Success convert LiDAR points into BEV tensor log {} frame number {} to file : {}".format( index_log_argoverse , frame_argoverse_index , name_of_lidar_frame_file ))
        print( "-----------------------------------------" )
        
        """

        # Makin tensor label for drivable area map

        drivable_area_label = argoverse_data.get_rasterized_drivabel_area_label( key = frame_argoverse_index )

        name_of_drivable_area_label = name_of_bev_drivable_area_label_folder + str( argoverse_data._lidar_list[ log_id ][ frame_argoverse_index ] ).split( "/")[-1].replace( ".ply" , "" ) + ".pickle"

        with open( name_of_drivable_area_label , 'wb+') as handle:
            pickle.dump( drivable_area_label, handle)

        print( "Sum of drivable area is : " + str( np.sum( drivable_area_label )))
        

        print( "Succes creating BEV drivable area label log {} frame number : {} to file : {}".format( index_log_argoverse , frame_argoverse_index , name_of_drivable_area_label ) )
        print( "-------------------------------------------")


    argoverse_loader.__next__()
        

            