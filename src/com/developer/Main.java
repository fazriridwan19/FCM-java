package com.developer;

import java.io.IOException;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        FCM cmean = new FCM("data_lulus_tepat_waktu.csv");

        //get number of class from user
        System.out.println("Please input number of cluster that you want :");
        Scanner sc= new Scanner(System.in);
        String read1 = sc.nextLine();

        //run clustering algorithm
        cmean.run(Integer.parseInt(read1), 100);


        System.out.println("Clustering Finished!!!");


    }
}