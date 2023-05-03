import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

const API = {
    pos: '/api/',
}

@Injectable()
export class AppService {
    constructor(private http: HttpClient) { }

    getpos(text: any) {
        return this.http.post(API.pos, {text: text});
        // return this.http.get(API.pos);
    }

}
